"""
quick_test.py — 5-epoch ablation test to verify each fix before full training.

Usage:
    python quick_test.py <data_dir> [--fix data|arcface|pretrained|all]

Expected EER targets per fix:
    Baseline  (no fixes)  : ~26%
    Fix 1: data split     : ~15%
    Fix 2: arcface config : ~8%
    Fix 3: pretrained     : ~4%
    Fix 4: full config    : ~2%
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from dataset_pairup import build_dataloaders
from feature_extraction import LogMelExtractor
from loss import ArcFaceLoss, ContrastiveLoss
from model import SpeakerVerificationModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    thresholds = np.linspace(scores.min(), scores.max(), 300)
    far = np.array([((scores >= t) & (labels == 0)).sum() / n_neg for t in thresholds])
    frr = np.array([((scores < t) & (labels == 1)).sum() / n_pos for t in thresholds])
    idx = np.abs(far - frr).argmin()
    return float((far[idx] + frr[idx]) / 2)


@torch.no_grad()
def eval_eer(model, loader, device) -> float:
    model.eval()
    scores, labels = [], []
    for batch in loader:
        a, b, _la, _lb, lbl, *_ = batch
        a, b = a.to(device), b.to(device)
        ea, eb = model(a, b)
        scores.extend(F.cosine_similarity(ea, eb).cpu().tolist())
        labels.extend(lbl.tolist())
    return compute_eer(np.array(scores), np.array(labels))


def run_quick_test(
    data_dir: str,
    fix: str = "all",
    quick_epochs: int = 5,
    target_eer: float = 0.20,
    seed: int = 42,
) -> dict[str, float]:
    """Run 5-epoch test with the specified fix level.

    Args:
        data_dir: Root speaker directory.
        fix: One of ``data | arcface | pretrained | all``.
        quick_epochs: Number of training epochs (default 5).
        target_eer: Pass threshold (EER below this is a pass).
        seed: Random seed.

    Returns:
        Dict with ``{'final_eer': float, 'passed': bool}``.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Config per fix level ─────────────────────────────────────────────────
    configs = {
        "baseline": dict(batch_size=16,  lr=1e-3, arcface_margin=0.3, arcface_scale=30.0,
                         use_pretrained=False),
        "data":     dict(batch_size=32,  lr=1e-3, arcface_margin=0.3, arcface_scale=30.0,
                         use_pretrained=False),
        "arcface":  dict(batch_size=32,  lr=1e-3, arcface_margin=0.5, arcface_scale=30.0,
                         use_pretrained=False),
        "pretrained": dict(batch_size=64, lr=1e-4, arcface_margin=0.5, arcface_scale=30.0,
                           use_pretrained=True),
        "all":      dict(batch_size=64,  lr=1e-4, arcface_margin=0.5, arcface_scale=30.0,
                         use_pretrained=True),
    }
    cfg = configs.get(fix, configs["all"])
    print(f"\n{'='*60}")
    print(f"  QUICK TEST — fix='{fix}'  ({quick_epochs} epochs)")
    print(f"  Config: {cfg}")
    print(f"{'='*60}")

    # ── Scan files ───────────────────────────────────────────────────────────
    file_list: list[tuple[str, str]] = []
    for spk_dir in sorted(Path(data_dir).iterdir()):
        if not spk_dir.is_dir():
            continue
        for ext in ("*.wav", "*.flac"):
            for f in spk_dir.glob(ext):
                file_list.append((str(f), spk_dir.name))

    if not file_list:
        print(f"[ERROR] No audio files found in {data_dir}")
        return {"final_eer": 0.5, "passed": False}

    num_spk = len(set(s for _, s in file_list))
    print(f"[data] {len(file_list)} files, {num_spk} speakers")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    extractor = LogMelExtractor()
    train_loader, val_loader, train_spk, val_spk = build_dataloaders(
        file_list, extractor,
        num_pairs=2000,               # smaller for quick test
        batch_size=cfg["batch_size"],
        num_workers=2,
        train_frac=0.8,
        seed=seed,
    )

    # ── Model & loss ─────────────────────────────────────────────────────────
    model = SpeakerVerificationModel(
        n_mels=80, channels=512, embedding_dim=128,
        use_pretrained=cfg["use_pretrained"],
    ).to(device)

    arcface = ArcFaceLoss(
        embedding_dim=128,
        num_classes=len(train_spk),
        margin_m=cfg["arcface_margin"],
        scale_s=cfg["arcface_scale"],
    ).to(device)

    contrastive = ContrastiveLoss()
    optimizer = AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=cfg["lr"], weight_decay=1e-4,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler("cuda")

    train_spk_sorted = sorted(train_spk)
    train_spk_to_idx = {s: i for i, s in enumerate(train_spk_sorted)}

    # ── Train ─────────────────────────────────────────────────────────────────
    eer_history = []
    for epoch in range(quick_epochs):
        train_loader.dataset._regenerate_pairs()   # type: ignore[union-attr]
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            a, b, _la, _lb, pair_labels, spk_a, spk_b = batch
            a, b = a.to(device), b.to(device)
            pair_float = pair_labels.float().to(device)
            lbl_a = torch.tensor([train_spk_to_idx.get(s, 0) for s in spk_a],
                                  dtype=torch.long, device=device)
            lbl_b = torch.tensor([train_spk_to_idx.get(s, 0) for s in spk_b],
                                  dtype=torch.long, device=device)

            optimizer.zero_grad()
            with autocast("cuda"):
                ea, eb = model(a, b)
                all_emb = torch.cat([ea, eb])
                all_lbl = torch.cat([lbl_a, lbl_b])
                loss = arcface(all_emb, all_lbl) + 0.1 * contrastive(ea, eb, pair_float)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()
        val_eer = eval_eer(model, val_loader, device)
        eer_history.append(val_eer)
        print(f"  Epoch {epoch+1}/{quick_epochs} — "
              f"loss={total_loss/len(train_loader):.4f}  val_eer={val_eer*100:.2f}%")

    final_eer = eer_history[-1]
    passed = final_eer < target_eer

    print(f"\n{'='*60}")
    print(f"  RESULT: final_eer = {final_eer*100:.2f}%  "
          f"(target < {target_eer*100:.0f}%)")
    print(f"  {'✅ PASS — proceed to full training' if passed else '❌ FAIL — apply next fix'}")
    print(f"{'='*60}\n")

    return {"final_eer": final_eer, "passed": passed}


# ---------------------------------------------------------------------------
# Ablation study: run all fix levels in sequence
# ---------------------------------------------------------------------------

def ablation_study(data_dir: str, quick_epochs: int = 5) -> None:
    """Run all fix levels and print an ablation table."""
    results: dict[str, dict] = {}
    fix_levels = ["baseline", "data", "arcface", "pretrained", "all"]
    targets = [0.30, 0.20, 0.12, 0.07, 0.05]

    for fix, target in zip(fix_levels, targets):
        r = run_quick_test(data_dir, fix=fix, quick_epochs=quick_epochs, target_eer=target)
        results[fix] = r

    print("\n" + "=" * 55)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 55)
    print(f"  {'Fix':<12}  {'EER (%)':<10}  {'Target':<10}  {'Status'}")
    print("  " + "-" * 50)
    prev_eer = None
    for fix, target in zip(fix_levels, targets):
        r = results[fix]
        eer_pct = r["final_eer"] * 100
        delta = f" ({(prev_eer - r['final_eer'])*100:+.1f}pp)" if prev_eer else ""
        status = "✅" if r["passed"] else "❌"
        print(f"  {fix:<12}  {eer_pct:<10.2f}  {target*100:<10.0f}  {status}{delta}")
        prev_eer = r["final_eer"]
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick 5-epoch test for each fix level")
    parser.add_argument("data_dir", help="Root: speaker_id/utterance.wav")
    parser.add_argument("--fix", default="all",
                        choices=["baseline", "data", "arcface", "pretrained", "all"],
                        help="Which fix level to test")
    parser.add_argument("--ablation", action="store_true",
                        help="Run full ablation study across all fix levels")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--target_eer", type=float, default=0.20,
                        help="EER threshold to pass (default 0.20 = 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.ablation:
        ablation_study(args.data_dir, quick_epochs=args.epochs)
    else:
        run_quick_test(
            args.data_dir,
            fix=args.fix,
            quick_epochs=args.epochs,
            target_eer=args.target_eer,
            seed=args.seed,
        )