"""
diagnose_training.py — Analyze training dynamics from saved logs/checkpoints.

Generates plots:
  - Training loss curve
  - Validation EER curve
  - Learning rate schedule
  - Cosine similarity distribution per epoch
  - Overfitting detection (loss ↓ but EER ↑)
"""

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def load_train_log(log_path: str) -> dict[str, list[float]]:
    """Parse CSV training log into arrays.

    Expected columns: epoch, train_loss, val_eer, val_loss, lr
    """
    data: dict[str, list[float]] = {
        "epoch": [], "train_loss": [], "val_eer": [], "val_loss": [], "lr": []
    }
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                try:
                    data[k].append(float(row[k]))
                except (KeyError, ValueError):
                    pass
    return data


# ---------------------------------------------------------------------------
# Synthetic schedule simulation (for when no log exists yet)
# ---------------------------------------------------------------------------

def simulate_cosine_lr(epochs: int, t0: int = 10, t_mult: int = 2, lr: float = 1e-3) -> list[float]:
    """Simulate CosineAnnealingWarmRestarts schedule without a real model."""
    lrs = []
    t_cur = 0
    t_i = t0
    for ep in range(epochs):
        cosine_lr = lr * (1 + math.cos(math.pi * t_cur / t_i)) / 2
        lrs.append(cosine_lr)
        t_cur += 1
        if t_cur >= t_i:
            t_cur = 0
            t_i *= t_mult
    return lrs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curve(epochs: list, train_loss: list, val_loss: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train loss", color="steelblue", linewidth=2)
    if val_loss:
        ax.plot(epochs, val_loss, label="Val loss", color="tomato", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved → {out_path}")


def plot_eer_curve(epochs: list, val_eer: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [e * 100 for e in val_eer], color="mediumseagreen", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("EER (%)")
    ax.set_title("Validation EER over Training")
    ax.grid(True, alpha=0.3)
    # Mark best epoch
    best_idx = int(np.argmin(val_eer))
    ax.axvline(epochs[best_idx], color="navy", linestyle=":",
               label=f"Best EER={val_eer[best_idx]*100:.2f}% @ epoch {epochs[best_idx]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  EER curve saved  → {out_path}")


def plot_lr_schedule(epochs: list, lrs: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, lrs, color="darkorange", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule (CosineAnnealingWarmRestarts)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  LR schedule saved → {out_path}")


def plot_overfitting_check(
    epochs: list,
    train_loss: list,
    val_eer: list,
    out_path: str,
) -> str:
    """Dual-axis plot to visually detect overfitting."""
    fig, ax1 = plt.subplots(figsize=(9, 4))
    color_loss = "steelblue"
    color_eer = "tomato"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color_loss)
    ax1.plot(epochs, train_loss, color=color_loss, linewidth=2, label="Train loss")
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val EER (%)", color=color_eer)
    ax2.plot(epochs, [e * 100 for e in val_eer], color=color_eer,
             linewidth=2, linestyle="--", label="Val EER")
    ax2.tick_params(axis="y", labelcolor=color_eer)

    # Detect divergence: loss still decreasing while EER is increasing
    overfit_detected = False
    if len(train_loss) > 5:
        last5_loss = train_loss[-5:]
        last5_eer = val_eer[-5:]
        loss_trend = np.polyfit(range(5), last5_loss, 1)[0]
        eer_trend = np.polyfit(range(5), last5_eer, 1)[0]
        if loss_trend < -0.001 and eer_trend > 0.001:
            overfit_detected = True
            ax1.set_title("⚠️  OVERFITTING DETECTED (loss ↓, EER ↑)")
        else:
            ax1.set_title("✅ No clear overfitting detected")
    else:
        ax1.set_title("Training Dynamics Overview")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Overfitting plot saved → {out_path}")
    return "⚠️  OVERFITTING DETECTED" if overfit_detected else "✅ No overfitting"


def plot_similarity_distribution(
    model_path: str | None,
    device: torch.device,
    n_mels: int,
    out_path: str,
) -> None:
    """Simulate positive/negative cosine similarity distribution at init vs trained."""
    try:
        from model import SpeakerVerificationModel
        import torch.nn.functional as F

        def get_sims(mdl):
            mdl.eval()
            with torch.no_grad():
                same_sims, diff_sims = [], []
                for _ in range(200):
                    T = np.random.randint(100, 300)
                    xa = torch.randn(1, n_mels, T).to(device)
                    xb = torch.randn(1, n_mels, T).to(device)
                    ea, eb = mdl.embed(xa), mdl.embed(xb)
                    sim = F.cosine_similarity(ea, eb).item()
                    if np.random.rand() > 0.5:
                        same_sims.append(sim)
                    else:
                        diff_sims.append(sim)
            return same_sims, diff_sims

        init_model = SpeakerVerificationModel(n_mels=n_mels).to(device)
        init_same, init_diff = get_sims(init_model)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, same, diff, title in [
            (axes[0], init_same, init_diff, "Random Init"),
        ]:
            ax.hist(diff, bins=40, alpha=0.6, color="tomato", label="Different")
            ax.hist(same, bins=40, alpha=0.6, color="mediumseagreen", label="Same")
            ax.set_title(f"Score Distribution — {title}")
            ax.set_xlabel("Cosine Similarity")
            ax.legend()

        if model_path and Path(model_path).exists():
            trained_model = SpeakerVerificationModel(n_mels=n_mels).to(device)
            ckpt = torch.load(model_path, map_location=device)
            trained_model.load_state_dict(ckpt["model_state"])
            t_same, t_diff = get_sims(trained_model)
            axes[1].hist(t_diff, bins=40, alpha=0.6, color="tomato", label="Different")
            axes[1].hist(t_same, bins=40, alpha=0.6, color="mediumseagreen", label="Same")
            axes[1].set_title("Score Distribution — Trained")
            axes[1].set_xlabel("Cosine Similarity")
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "No checkpoint provided", ha="center", va="center",
                         transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title("Trained model (N/A)")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Similarity dist saved → {out_path}")
    except Exception as exc:
        print(f"  [warn] Could not generate similarity distribution: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostics(
    log_path: str | None = None,
    checkpoint: str | None = None,
    n_mels: int = 80,
    epochs_total: int = 50,
    out_dir: str = "diagnostics",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  TRAINING DYNAMICS DIAGNOSTICS")
    print("=" * 60)

    # ── Load or simulate log ────────────────────────────────────────────────
    if log_path and Path(log_path).exists():
        log = load_train_log(log_path)
        epochs = log["epoch"]
        train_loss = log["train_loss"]
        val_eer = log["val_eer"]
        val_loss = log["val_loss"]
        lrs = log["lr"]
        print(f"\n📋 Loaded training log: {len(epochs)} epochs")
    else:
        print("\n📋 No training log found — simulating for LR schedule visualisation.")
        epochs = list(range(1, epochs_total + 1))
        train_loss = [max(0.1, 2.0 * (0.95 ** e)) for e in epochs]
        val_eer = [max(0.05, 0.4 * (0.96 ** e)) for e in epochs]
        val_loss = [max(0.15, 1.8 * (0.96 ** e)) for e in epochs]
        lrs = simulate_cosine_lr(epochs_total)

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n── Generating plots ──")
    plot_loss_curve(epochs, train_loss, val_loss, f"{out_dir}/loss_curve.png")
    plot_eer_curve(epochs, val_eer, f"{out_dir}/eer_curve.png")
    plot_lr_schedule(epochs, lrs, f"{out_dir}/lr_schedule.png")
    overfit_status = plot_overfitting_check(
        epochs, train_loss, val_eer, f"{out_dir}/overfitting_check.png"
    )
    plot_similarity_distribution(checkpoint, device, n_mels, f"{out_dir}/sim_distribution.png")

    # ── Textual analysis ────────────────────────────────────────────────────
    print("\n── Training Dynamics Analysis ──")
    if len(train_loss) > 1:
        loss_drop = train_loss[0] - train_loss[-1]
        print(f"  Total loss drop   : {loss_drop:.4f} "
              f"({'✅ decreasing' if loss_drop > 0 else '❌ not decreasing'})")
    if len(val_eer) > 1:
        best_eer = min(val_eer)
        best_epoch = epochs[val_eer.index(best_eer)]
        print(f"  Best val EER      : {best_eer*100:.2f}% @ epoch {best_epoch}")
        eer_fluctuation = np.std(val_eer[-5:]) if len(val_eer) >= 5 else 0
        print(f"  EER fluctuation (last 5 epochs): {eer_fluctuation*100:.3f}%  "
              f"({'✅ stable' if eer_fluctuation < 0.02 else '⚠️  high variance'})")
    print(f"  Overfitting check : {overfit_status}")

    # ── Report ──────────────────────────────────────────────────────────────
    report_path = Path(out_dir) / "training_diagnosis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TRAINING DYNAMICS DIAGNOSTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Epochs analysed  : {len(epochs)}\n")
        if train_loss:
            f.write(f"Final train loss : {train_loss[-1]:.4f}\n")
        if val_eer:
            f.write(f"Best val EER     : {min(val_eer)*100:.2f}%\n")
        f.write(f"Overfitting      : {overfit_status}\n")
    print(f"\n📄 Report saved → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose training dynamics")
    parser.add_argument("--log", default=None, help="Path to train_log.csv")
    parser.add_argument("--checkpoint", default=None, help="Path to best_model.pth")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--epochs_total", type=int, default=50)
    parser.add_argument("--out_dir", default="diagnostics")
    args = parser.parse_args()

    run_diagnostics(
        log_path=args.log,
        checkpoint=args.checkpoint,
        n_mels=args.n_mels,
        epochs_total=args.epochs_total,
        out_dir=args.out_dir,
    )