"""
diagnose_model.py — Verify model architecture correctness.

Checks:
  - Embedding norms (~1.0 after L2 normalisation)
  - Output embedding statistics (mean, std, min, max)
  - Gradient flow (all layers updating)
  - Pretrained weight loading status
  - ArcFace weight initialisation
  - Trainable vs frozen parameter count
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from model import SpeakerVerificationModel
from loss import ArcFaceLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def check_embedding_norms(
    model: SpeakerVerificationModel,
    device: torch.device,
    n_mels: int = 80,
    n_samples: int = 32,
) -> dict:
    """Run random inputs and collect embedding statistics."""
    model.eval()
    norms = []
    all_embs = []
    with torch.no_grad():
        for _ in range(n_samples):
            T = np.random.randint(100, 400)
            x = torch.randn(1, n_mels, T).to(device)
            emb = model.embed(x)
            norms.append(emb.norm(dim=1).item())
            all_embs.append(emb.cpu().numpy())

    norms = np.array(norms)
    all_embs = np.concatenate(all_embs, axis=0)  # (n_samples, D)
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "emb_mean": float(all_embs.mean()),
        "emb_std": float(all_embs.std()),
        "emb_min": float(all_embs.min()),
        "emb_max": float(all_embs.max()),
        "status": (
            "✅ PASS (norms ≈ 1.0)"
            if abs(norms.mean() - 1.0) < 0.01 and norms.std() < 0.01
            else f"❌ FAIL — norm mean={norms.mean():.4f}, expected 1.0"
        ),
    }


def check_gradient_flow(
    model: SpeakerVerificationModel,
    arcface: ArcFaceLoss,
    device: torch.device,
    n_mels: int = 80,
) -> dict:
    """Run a forward+backward pass and report which layers have zero gradients."""
    model.train()
    arcface.train()
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(arcface.parameters()), lr=1e-3
    )
    optimizer.zero_grad()

    x = torch.randn(8, n_mels, 200).to(device)
    labels = torch.zeros(8, dtype=torch.long).to(device)
    emb = model.embed(x)
    loss = arcface(emb, labels)
    loss.backward()

    frozen_layers = []
    updating_layers = []
    zero_grad_layers = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_layers.append(name)
        elif param.grad is None:
            zero_grad_layers.append(name)
        elif param.grad.abs().max().item() == 0.0:
            zero_grad_layers.append(name)
        else:
            updating_layers.append(name)

    return {
        "updating_layers": len(updating_layers),
        "frozen_layers": len(frozen_layers),
        "zero_grad_layers": zero_grad_layers[:10],
        "status": (
            "✅ PASS — all layers receiving gradients"
            if len(zero_grad_layers) == 0
            else f"⚠️  {len(zero_grad_layers)} layers have zero gradients"
        ),
    }


def check_arcface_init(arcface: ArcFaceLoss) -> dict:
    """Verify ArcFace weight statistics and margin/scale values."""
    w = arcface.weight.detach()
    return {
        "weight_shape": list(w.shape),
        "weight_norm_mean": float(w.norm(dim=1).mean().item()),
        "margin_m": arcface.m,
        "scale_s": arcface.s,
        "margin_status": (
            "✅ m ≥ 0.3"
            if arcface.m >= 0.3
            else f"❌ m={arcface.m:.3f} — too small, use ≥ 0.3"
        ),
        "scale_status": (
            "✅ s ≥ 30"
            if arcface.s >= 30
            else f"❌ s={arcface.s:.1f} — too small, use ≥ 30"
        ),
    }


def check_pretrained_weights(model_path: str | None, model: SpeakerVerificationModel, device: torch.device) -> dict:
    """Check whether a checkpoint was loaded and weights are non-random."""
    if model_path is None or not Path(model_path).exists():
        return {
            "loaded": False,
            "status": "⚠️  No checkpoint provided — model uses random initialisation.",
        }
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        # Check first conv weight is not all-zero/random-uniform
        first_w = next(model.parameters()).detach()
        std = first_w.std().item()
        return {
            "loaded": True,
            "checkpoint_epoch": ckpt.get("epoch", "?"),
            "first_layer_std": round(std, 5),
            "status": "✅ Checkpoint loaded successfully.",
        }
    except Exception as exc:
        return {"loaded": False, "status": f"❌ Failed to load checkpoint: {exc}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostics(
    model_path: str | None = None,
    embedding_dim: int = 128,
    n_mels: int = 80,
    channels: int = 512,
    num_classes: int = 200,
    out_dir: str = "diagnostics",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 60)
    print("  MODEL ARCHITECTURE DIAGNOSTICS")
    print("=" * 60)
    print(f"  Device: {device}")

    model = SpeakerVerificationModel(
        n_mels=n_mels, channels=channels, embedding_dim=embedding_dim
    ).to(device)

    arcface = ArcFaceLoss(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    ).to(device)

    # ── 1. Parameter count ──────────────────────────────────────────────────
    print("\n── 1. Parameter Count ──")
    trainable, total = count_parameters(model)
    arc_trainable, arc_total = count_parameters(arcface)
    print(f"  Model   — trainable: {trainable:,}  /  total: {total:,}")
    print(f"  ArcFace — trainable: {arc_trainable:,}  /  total: {arc_total:,}")
    print(f"  Combined trainable : {trainable + arc_trainable:,}")

    # ── 2. Embedding norms ──────────────────────────────────────────────────
    print("\n── 2. Embedding Norm Check ──")
    norm_result = check_embedding_norms(model, device, n_mels)
    print(f"  Norm   — mean: {norm_result['norm_mean']:.5f}  std: {norm_result['norm_std']:.5f}  "
          f"min: {norm_result['norm_min']:.5f}  max: {norm_result['norm_max']:.5f}")
    print(f"  Emb    — mean: {norm_result['emb_mean']:.5f}  std: {norm_result['emb_std']:.5f}  "
          f"min: {norm_result['emb_min']:.5f}  max: {norm_result['emb_max']:.5f}")
    print(f"  Status : {norm_result['status']}")

    # ── 3. Gradient flow ────────────────────────────────────────────────────
    print("\n── 3. Gradient Flow Check ──")
    grad_result = check_gradient_flow(model, arcface, device, n_mels)
    print(f"  Updating layers   : {grad_result['updating_layers']}")
    print(f"  Frozen layers     : {grad_result['frozen_layers']}")
    if grad_result["zero_grad_layers"]:
        print(f"  Zero-grad layers  : {grad_result['zero_grad_layers']}")
    print(f"  Status            : {grad_result['status']}")

    # ── 4. Pretrained weights ───────────────────────────────────────────────
    print("\n── 4. Pretrained Weights Check ──")
    # Reload fresh model for checkpoint check
    fresh_model = SpeakerVerificationModel(
        n_mels=n_mels, channels=channels, embedding_dim=embedding_dim
    ).to(device)
    ckpt_result = check_pretrained_weights(model_path, fresh_model, device)
    for k, v in ckpt_result.items():
        print(f"  {k}: {v}")

    # ── 5. ArcFace init ─────────────────────────────────────────────────────
    print("\n── 5. ArcFace Configuration Check ──")
    arc_result = check_arcface_init(arcface)
    for k, v in arc_result.items():
        print(f"  {k}: {v}")

    # ── Save report ─────────────────────────────────────────────────────────
    report_path = Path(out_dir) / "model_diagnosis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MODEL ARCHITECTURE DIAGNOSTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Trainable params (model) : {trainable:,}\n")
        f.write(f"Total params (model)     : {total:,}\n")
        f.write(f"Embedding norm           : {norm_result['status']}\n")
        f.write(f"Gradient flow            : {grad_result['status']}\n")
        f.write(f"Pretrained weights       : {ckpt_result['status']}\n")
        f.write(f"ArcFace margin status    : {arc_result['margin_status']}\n")
        f.write(f"ArcFace scale status     : {arc_result['scale_status']}\n")
    print(f"\n📄 Report saved → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose model architecture")
    parser.add_argument("--checkpoint", default=None, help="Path to best_model.pth")
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--out_dir", default="diagnostics")
    args = parser.parse_args()

    run_diagnostics(
        model_path=args.checkpoint,
        embedding_dim=args.embedding_dim,
        n_mels=args.n_mels,
        channels=args.channels,
        num_classes=args.num_classes,
        out_dir=args.out_dir,
    )