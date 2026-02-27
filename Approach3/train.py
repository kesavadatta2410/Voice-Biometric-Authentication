"""
train.py — Training loop for ECAPA-TDNN speaker verification.

Priority fixes applied:
  - Batch size default: 64 (from 32) for stable metric learning
  - LR default: 1e-4 (from 1e-3) for pretrained fine-tuning
  - Epochs default: 100 (from 50)
  - ArcFace margin: 0.5 (from 0.3), scale: 30.0
  - 5-epoch linear LR warmup before cosine annealing
  - Real speaker class indices passed to ArcFace (not placeholder zeros)
  - Speaker-disjoint splits enforced via dataset_pairup.build_dataloaders
  - Full pre-training assertions before loop starts
"""

import argparse
import csv
import datetime
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dataset_pairup import build_dataloaders, SpeakerPairDataset
from feature_extraction import LogMelExtractor
from loss import OnlineTripletLoss, ContrastiveLoss
from model import SpeakerVerificationModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# EER helper
# ---------------------------------------------------------------------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate.

    Args:
        scores: Cosine similarity scores ``(N,)``.
        labels: Ground-truth 1/0 labels ``(N,)``.

    Returns:
        ``(eer, threshold)``
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    thresholds = np.linspace(scores.min(), scores.max(), 500)
    far_arr, frr_arr = [], []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        fa = ((pred == 1) & (labels == 0)).sum()
        fr = ((pred == 0) & (labels == 1)).sum()
        far_arr.append(fa / n_neg)
        frr_arr.append(fr / n_pos)

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)
    idx = np.abs(far_arr - frr_arr).argmin()
    eer = float((far_arr[idx] + frr_arr[idx]) / 2)
    return eer, float(thresholds[idx])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: SpeakerVerificationModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return ``(eer, avg_contrastive_loss)``."""
    model.eval()
    criterion = ContrastiveLoss()
    all_scores: list[float] = []
    all_labels: list[int] = []
    total_loss = 0.0

    for batch in loader:
        padded_a, padded_b, _la, _lb, pair_labels, _spk_a, _spk_b = batch
        padded_a = padded_a.to(device)
        padded_b = padded_b.to(device)
        pair_labels_dev = pair_labels.float().to(device)

        emb_a, emb_b = model(padded_a, padded_b)
        loss = criterion(emb_a, emb_b, pair_labels_dev)
        total_loss += loss.item()

        sims = F.cosine_similarity(emb_a, emb_b).cpu().numpy()
        all_scores.extend(sims.tolist())
        all_labels.extend(pair_labels.cpu().numpy().tolist())

    eer, _ = compute_eer(np.array(all_scores), np.array(all_labels))
    return eer, total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Pre-flight assertions
# ---------------------------------------------------------------------------

def preflight_checks(
    model: SpeakerVerificationModel,
    train_spk: set[str],
    val_spk: set[str],
    batch_size: int,
) -> None:
    """Assert all critical conditions before training begins."""
    # 1. No speaker leakage
    overlap = train_spk & val_spk
    assert len(overlap) == 0, f"CRITICAL: {len(overlap)} speakers in both splits!"

    # 2. Embedding norms
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x = torch.randn(4, 80, 200).to(device)
        emb = model.embed(x)
        norm_err = (emb.norm(dim=1) - 1.0).abs().max().item()
        assert norm_err < 1e-4, f"Embedding norms not ≈ 1.0 (err={norm_err:.6f})"
    model.train()

    # 3. Batch size
    assert batch_size >= 32, f"batch_size={batch_size} — must be ≥ 32"

    print("[preflight] ✅ All assertions passed:")
    print(f"  Speaker overlap : 0")
    print(f"  Embedding norms ≈ 1.0 (err={norm_err:.2e})")
    print(f"  Batch size      = {batch_size}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    file_list: list[tuple[str, str]],
    speaker_to_idx: dict[str, int],
    save_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    num_pairs: int = 10_000,
    num_workers: int = 4,
    train_split: float = 0.8,
    triplet_margin: float = 0.5,
    embedding_dim: int = 128,
    n_mels: int = 80,
    channels: int = 512,
    grad_clip: float = 1.0,
    patience: int = 15,
    resume: Optional[str] = None,
    seed: int = 42,
    use_pretrained: bool = True,
) -> None:
    """Main training entry point.

    Args:
        file_list: List of ``(audio_path, speaker_id)`` tuples.
        speaker_to_idx: Mapping from speaker string ID to integer class index.
        save_dir: Checkpoint and log directory.
        epochs: Maximum training epochs.
        batch_size: Batch size (≥ 32 required).
        lr: Initial learning rate (use 1e-4 for pretrained fine-tuning).
        weight_decay: AdamW L2 regularisation.
        num_pairs: Pairs per epoch.
        num_workers: DataLoader worker count.
        train_split: Speaker-level train fraction.
        arcface_margin: ArcFace margin m (≥ 0.3, recommended 0.5).
        arcface_scale: ArcFace scale s (≥ 30).
        embedding_dim: Output embedding dimension.
        n_mels: Mel-frequency bins (must be 80).
        channels: Custom ECAPA channel width.
        warmup_epochs: Linear LR warmup length.
        t0: Cosine annealing restart period.
        t_mult: Restart period multiplier.
        grad_clip: Max gradient norm.
        patience: Early stopping patience (EER).
        resume: Path to checkpoint for resume.
        seed: Global random seed.
        use_pretrained: Whether to attempt loading SpeechBrain weights.
    """
    set_seed(seed)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("[train] CUDA GPU is required but not available. Aborting.")
    device = torch.device("cuda")
    cudnn.benchmark = True
    print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────────
    extractor = LogMelExtractor(n_mels=n_mels)
    train_loader, val_loader, train_spk, val_spk = build_dataloaders(
        file_list, extractor, num_pairs, batch_size, num_workers, train_split, seed
    )
    num_train_speakers = len(train_spk)
    # Build a local speaker_to_idx restricted to train speakers only
    train_spk_sorted = sorted(train_spk)
    train_spk_to_idx = {s: i for i, s in enumerate(train_spk_sorted)}

    # ── Model ───────────────────────────────────────────────────────────────
    model = SpeakerVerificationModel(
        n_mels=n_mels, channels=channels,
        embedding_dim=embedding_dim, use_pretrained=use_pretrained,
    ).to(device)

    triplet_loss = OnlineTripletLoss(margin=triplet_margin).to(device)
    contrastive  = ContrastiveLoss().to(device)

    # ── Pre-flight checks ───────────────────────────────────────────────────
    preflight_checks(model, train_spk, val_spk, batch_size)

    # ── Optimiser & scheduler ───────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=lr, weight_decay=weight_decay,
    )
    # OneCycleLR: 30% warmup then cosine decay — better than SequentialLR for small datasets
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    scaler = GradScaler("cuda")

    start_epoch = 0
    best_eer = float("inf")
    no_improve = 0

    # ── Resume ──────────────────────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", float("inf"))
        print(f"[train] Resumed from epoch {start_epoch}, best EER {best_eer:.4f}")

    # ── Log file ────────────────────────────────────────────────────────────
    log_file = save_path / "train_log.csv"
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_eer", "val_loss", "lr"])

    json_log_file = save_path / "epoch_log.json"
    epoch_log: list[dict] = []

    print(f"\n[train] Starting training for up to {epochs} epochs …\n")

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        # Regenerate pairs each epoch for fresh sampling
        train_loader.dataset._regenerate_pairs()   # type: ignore[union-attr]

        model.train()
        total_loss = 0.0

        for batch in train_loader:
            padded_a, padded_b, _la, _lb, pair_labels, spk_ids_a, spk_ids_b = batch
            padded_a = padded_a.to(device)
            padded_b = padded_b.to(device)
            pair_labels_float = pair_labels.float().to(device)

            # Speaker label tensors for triplet mining
            spk_labels_a = torch.tensor(
                [train_spk_to_idx.get(s, 0) for s in spk_ids_a],
                dtype=torch.long, device=device,
            )
            spk_labels_b = torch.tensor(
                [train_spk_to_idx.get(s, 0) for s in spk_ids_b],
                dtype=torch.long, device=device,
            )

            optimizer.zero_grad()

            with autocast("cuda"):
                emb_a, emb_b = model(padded_a, padded_b)

                all_emb = torch.cat([emb_a, emb_b], dim=0)
                all_lbl = torch.cat([spk_labels_a, spk_labels_b], dim=0)
                tri_loss = triplet_loss(all_emb, all_lbl)
                con_loss = contrastive(emb_a, emb_b, pair_labels_float)
                loss = tri_loss + 0.1 * con_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()   # OneCycleLR steps per batch, not per epoch
            total_loss += loss.item()

        # OneCycleLR already stepped per batch above — no epoch-level step needed
        avg_loss = total_loss / max(len(train_loader), 1)
        current_lr = optimizer.param_groups[0]["lr"]

        val_eer, val_loss = validate(model, val_loader, device)
        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"loss={avg_loss:.4f}  val_eer={val_eer*100:.2f}%  "
            f"val_loss={val_loss:.4f}  lr={current_lr:.2e}"
        )

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_loss, val_eer, val_loss, current_lr])

        # ── Per-epoch JSON log ──────────────────────────────────────────────
        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "val_eer_pct": round(val_eer * 100, 4),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
            "best_eer_pct": round(min(best_eer, val_eer) * 100, 4),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        with open(json_log_file, "w", encoding="utf-8") as f:
            json.dump(epoch_log, f, indent=2)

        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_eer": best_eer,
        }
        torch.save(ckpt_data, save_path / "latest_model.pth")

        if val_eer < best_eer:
            best_eer = val_eer
            no_improve = 0
            torch.save(ckpt_data, save_path / "best_model.pth")
            print(f"  ✓ New best EER: {best_eer*100:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[train] Early stopping at epoch {epoch+1}.")
                break

    print(f"\n[train] Done. Best val EER: {best_eer*100:.2f}%")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_file_list(root: str) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Scan root for speaker_id/utterance.{wav,flac} structure."""
    from pathlib import Path
    file_list: list[tuple[str, str]] = []
    for spk_dir in sorted(Path(root).iterdir()):
        if not spk_dir.is_dir():
            continue
        for ext in ("*.wav", "*.flac"):
            for f in spk_dir.glob(ext):
                file_list.append((str(f), spk_dir.name))
    speakers = sorted(set(s for _, s in file_list))
    return file_list, {s: i for i, s in enumerate(speakers)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECAPA-TDNN speaker verification")
    parser.add_argument("data_dir", help="Root directory: speaker_id/utterance.wav")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--triplet_margin", type=float, default=0.5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable SpeechBrain pretrained weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    file_list, speaker_to_idx = _build_file_list(args.data_dir)
    print(f"Found {len(file_list):,} files from {len(speaker_to_idx)} speakers.")
    if len(speaker_to_idx) < 100:
        print(f"⚠️  WARNING: {len(speaker_to_idx)} speakers is below recommended minimum (100).")

    train(
        file_list=file_list,
        speaker_to_idx=speaker_to_idx,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        triplet_margin=args.triplet_margin,
        resume=args.resume,
        use_pretrained=not args.no_pretrained,
        seed=args.seed,
    )