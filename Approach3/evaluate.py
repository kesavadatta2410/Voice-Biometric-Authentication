"""
evaluate.py — Comprehensive speaker verification evaluation.

Outputs (saved to results/):
  - eer_report.txt            : EER, minDCF, optimal threshold (human-readable)
  - results.json              : Same metrics in machine-readable JSON (always up-to-date)
  - det_curve.png             : Detection Error Tradeoff curve
  - similarity_distribution.png : Score histogram by class
  - tsne_embeddings.png       : t-SNE of speaker embeddings
"""

import argparse
import datetime
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from feature_extraction import LogMelExtractor, extract_from_file
from model import SpeakerVerificationModel
from preprocess import TARGET_SR


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_scores(
    trial_pairs: list[tuple[str, str, int]],
    model: SpeakerVerificationModel,
    extractor: LogMelExtractor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and compute cosine similarity scores.

    Args:
        trial_pairs: List of ``(path_a, path_b, label)`` tuples.
        model: Trained :class:`SpeakerVerificationModel`.
        extractor: Feature extractor.
        device: Compute device.

    Returns:
        ``(scores, labels)`` numpy arrays of shape ``(N,)``.
    """
    model.eval()
    scores: list[float] = []
    labels: list[int] = []

    for path_a, path_b, label in trial_pairs:
        fa = extract_from_file(path_a, extractor).squeeze(0).unsqueeze(0).to(device)  # (1, n_mels, T)
        fb = extract_from_file(path_b, extractor).squeeze(0).unsqueeze(0).to(device)
        emb_a = model.embed(fa)
        emb_b = model.embed(fb)
        sim = F.cosine_similarity(emb_a, emb_b).item()
        scores.append(sim)
        labels.append(label)

    return np.array(scores), np.array(labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_eer(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute EER and FAR/FRR curves.

    Args:
        scores: Cosine similarity scores.
        labels: Ground-truth 1/0.

    Returns:
        ``(eer, threshold, far_array, frr_array)``
    """
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    far_arr = []
    frr_arr = []

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        fa = ((pred == 1) & (labels == 0)).sum()
        fr = ((pred == 0) & (labels == 1)).sum()
        far_arr.append(fa / max(n_neg, 1))
        frr_arr.append(fr / max(n_pos, 1))

    far_arr = np.array(far_arr)
    frr_arr = np.array(frr_arr)
    diff = np.abs(far_arr - frr_arr)
    idx = diff.argmin()
    eer = float((far_arr[idx] + frr_arr[idx]) / 2)
    return eer, float(thresholds[idx]), far_arr, frr_arr


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """Compute minimum Detection Cost Function (minDCF).

    Args:
        scores: Cosine similarity scores.
        labels: Ground-truth 1/0.
        p_target: Prior probability of target.
        c_miss: Miss cost.
        c_fa: False alarm cost.

    Returns:
        Scalar minDCF value.
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    dcf_values = []

    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        pmiss = ((pred == 0) & (labels == 1)).sum() / max(n_pos, 1)
        pfa = ((pred == 1) & (labels == 0)).sum() / max(n_neg, 1)
        dcf = c_miss * p_target * pmiss + c_fa * (1 - p_target) * pfa
        dcf_values.append(dcf)

    # Normalise by the cost of the trivial system
    norm = min(c_miss * p_target, c_fa * (1 - p_target))
    return float(min(dcf_values) / norm)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_det_curve(
    far: np.ndarray, frr: np.ndarray, eer: float, out_path: str
) -> None:
    """Save a DET curve plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far * 100, frr * 100, color="steelblue", linewidth=2)
    ax.plot([eer * 100], [eer * 100], "ro", markersize=8, label=f"EER = {eer*100:.2f}%")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("Detection Error Tradeoff (DET) Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] DET curve saved → {out_path}")


def plot_score_distribution(
    scores: np.ndarray, labels: np.ndarray, threshold: float, out_path: str
) -> None:
    """Save a score histogram (same vs different speakers)."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(neg_scores, bins=80, alpha=0.6, color="tomato", label="Different speaker")
    ax.hist(pos_scores, bins=80, alpha=0.6, color="mediumseagreen", label="Same speaker")
    ax.axvline(threshold, color="navy", linestyle="--", linewidth=1.5,
               label=f"EER threshold = {threshold:.3f}")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution — Same vs Different Speaker")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Score distribution saved → {out_path}")


def plot_tsne(
    embeddings: np.ndarray,
    speaker_ids: list[str],
    out_path: str,
    max_speakers: int = 30,
    seed: int = 42,
) -> None:
    """Save a t-SNE visualisation of speaker embeddings.

    Args:
        embeddings: ``(N, D)`` array of L2-normalised embeddings.
        speaker_ids: Speaker label for each row.
        out_path: Output file path.
        max_speakers: Cap the number of speakers shown for clarity.
        seed: t-SNE random seed.
    """
    unique_speakers = list(set(speaker_ids))
    if len(unique_speakers) > max_speakers:
        unique_speakers = unique_speakers[:max_speakers]
        mask = [i for i, s in enumerate(speaker_ids) if s in unique_speakers]
        embeddings = embeddings[mask]
        speaker_ids = [speaker_ids[i] for i in mask]

    spk_to_int = {s: i for i, s in enumerate(unique_speakers)}
    int_labels = np.array([spk_to_int[s] for s in speaker_ids])

    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, spk in enumerate(unique_speakers):
        mask = int_labels == idx
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10,
                   color=cmap(idx % 20), label=spk, alpha=0.7)

    ax.set_title("t-SNE of Speaker Embeddings")
    ax.axis("off")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=cmap(i % 20), markersize=7)
               for i in range(len(unique_speakers))]
    if len(unique_speakers) <= 20:
        ax.legend(handles, unique_speakers, loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eval] t-SNE plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    trial_pairs: list[tuple[str, str, int]],
    model_path: str,
    results_dir: str = "results",
    n_mels: int = 80,
    channels: int = 512,
    embedding_dim: int = 128,
    p_target: float = 0.01,
    seed: int = 42,
) -> None:
    """Run full evaluation and save all result artefacts.

    Args:
        trial_pairs: ``(path_a, path_b, label)`` list.  Speaker-disjoint from train.
        model_path: Path to a saved ``best_model.pth`` checkpoint.
        results_dir: Directory for output files.
        n_mels: Mel-frequency bins.
        channels: ECAPA channel width.
        embedding_dim: Embedding dimension.
        p_target: minDCF prior target probability.
        seed: Random seed.
    """
    set_seed(seed)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Approach3 requires a GPU. "
            "Make sure torch+cu124 is installed and a CUDA GPU is available."
        )
    device = torch.device("cuda")
    print(f"[eval] Device: {torch.cuda.get_device_name(0)}")

    # Load model
    model = SpeakerVerificationModel(n_mels=n_mels, channels=channels, embedding_dim=embedding_dim)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[eval] Loaded model from {model_path}")

    extractor = LogMelExtractor(n_mels=n_mels)

    # ---- Scores ----
    scores, labels = compute_scores(trial_pairs, model, extractor, device)

    # ---- EER & minDCF ----
    eer, threshold, far, frr = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(scores, labels, p_target=p_target)

    print(f"[eval] EER       : {eer*100:.3f}%")
    print(f"[eval] minDCF    : {min_dcf:.4f}  (p_target={p_target})")
    print(f"[eval] Threshold : {threshold:.4f}")

    # ---- Report ----
    report_path = out_dir / "eer_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("  ECAPA-TDNN Speaker Verification — Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"  EER           : {eer*100:.3f}%\n")
        f.write(f"  minDCF        : {min_dcf:.4f}  (p_target={p_target})\n")
        f.write(f"  Threshold@EER : {threshold:.4f}\n")
        f.write(f"  Num trials    : {len(trial_pairs)}\n")
        f.write(f"  Checkpoint    : {model_path}\n")
        f.write("=" * 50 + "\n")
    print(f"[eval] Report saved → {report_path}")

    # ---- JSON results (always overwrites — fixes stale report issue) ----
    results = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(model_path),
        "num_trials": len(trial_pairs),
        "eer_pct": round(eer * 100, 4),
        "min_dcf": round(min_dcf, 6),
        "threshold_at_eer": round(float(threshold), 6),
        "p_target": p_target,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] JSON results  → {json_path}")

    # ---- DET curve ----
    plot_det_curve(far, frr, eer, str(out_dir / "det_curve.png"))

    # ---- Score distribution ----
    plot_score_distribution(scores, labels, threshold, str(out_dir / "similarity_distribution.png"))

    # ---- t-SNE: collect per-utterance embeddings ----
    print("[eval] Computing embeddings for t-SNE …")
    tsne_embeddings: list[np.ndarray] = []
    tsne_speakers: list[str] = []
    # Use unique files from trial_pairs
    seen: set[str] = set()
    for path_a, path_b, _ in trial_pairs:
        for path, spk in [(path_a, "?"), (path_b, "?")]:
            if path in seen:
                continue
            seen.add(path)
            try:
                feat = extract_from_file(path, extractor).squeeze(0).unsqueeze(0).to(device)
                emb = model.embed(feat).squeeze(0).cpu().numpy()
                tsne_embeddings.append(emb)
                # Derive speaker from parent directory name
                tsne_speakers.append(Path(path).parent.name)
            except Exception as exc:
                print(f"  [warn] Skipping {path}: {exc}")

    if len(tsne_embeddings) > 10:
        plot_tsne(
            np.stack(tsne_embeddings, axis=0),
            tsne_speakers,
            str(out_dir / "tsne_embeddings.png"),
            seed=seed,
        )
    else:
        print("[eval] Not enough embeddings for t-SNE — skipping.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ECAPA-TDNN speaker verification")
    parser.add_argument("trials_file",
                        help="TSV file: path_a<TAB>path_b<TAB>label (1=same, 0=diff)")
    parser.add_argument("model_path", help="Path to best_model.pth checkpoint")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--p_target", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse trials file
    trial_pairs: list[tuple[str, str, int]] = []
    with open(args.trials_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                trial_pairs.append((parts[0], parts[1], int(parts[2])))

    print(f"[eval] Loaded {len(trial_pairs)} trial pairs.")

    evaluate(
        trial_pairs=trial_pairs,
        model_path=args.model_path,
        results_dir=args.results_dir,
        n_mels=args.n_mels,
        channels=args.channels,
        embedding_dim=args.embedding_dim,
        p_target=args.p_target,
        seed=args.seed,
    )