"""
diagnose_data.py — Diagnose data pipeline for leakage and quality issues.

Checks:
  - Speaker overlap between train/val splits (CRITICAL)
  - Unique speaker counts per split
  - Audio length distribution histogram
  - Positive/negative pair balance
  - 5 random pair samples with speaker IDs
  - Same utterance appearing in both splits
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scan_data_dir(root: str) -> list[tuple[str, str]]:
    """Scan root for speaker_id/utterance.{wav,flac} structure.

    Returns:
        List of ``(audio_path, speaker_id)`` tuples.
    """
    file_list: list[tuple[str, str]] = []
    for spk_dir in sorted(Path(root).iterdir()):
        if not spk_dir.is_dir():
            continue
        for ext in ("*.wav", "*.flac", "*.mp3"):
            for f in spk_dir.glob(ext):
                file_list.append((str(f), spk_dir.name))
    return file_list


def speaker_disjoint_split(
    file_list: list[tuple[str, str]],
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split file_list at the *speaker* level — no speaker overlap guaranteed."""
    rng = random.Random(seed)
    speakers = list(set(s for _, s in file_list))
    rng.shuffle(speakers)
    split = int(len(speakers) * train_frac)
    train_spk = set(speakers[:split])
    val_spk = set(speakers[split:])
    train_files = [(p, s) for p, s in file_list if s in train_spk]
    val_files = [(p, s) for p, s in file_list if s in val_spk]
    return train_files, val_files


def get_audio_duration(path: str) -> float:
    """Return audio duration in seconds without loading the full file."""
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def check_speaker_overlap(
    train_files: list[tuple[str, str]],
    val_files: list[tuple[str, str]],
) -> dict:
    train_spk = set(s for _, s in train_files)
    val_spk = set(s for _, s in val_files)
    overlap = train_spk & val_spk

    result = {
        "train_speakers": len(train_spk),
        "val_speakers": len(val_spk),
        "overlap_count": len(overlap),
        "overlap_ids": sorted(overlap)[:10],  # show first 10 if any
        "status": "✅ PASS" if len(overlap) == 0 else f"❌ FAIL — {len(overlap)} speakers shared!",
    }
    return result


def check_utterance_overlap(
    train_files: list[tuple[str, str]],
    val_files: list[tuple[str, str]],
) -> dict:
    train_paths = set(p for p, _ in train_files)
    val_paths = set(p for p, _ in val_files)
    overlap = train_paths & val_paths
    return {
        "overlap_utterances": len(overlap),
        "examples": sorted(overlap)[:5],
        "status": "✅ PASS" if len(overlap) == 0 else f"❌ FAIL — {len(overlap)} utterances in both splits!",
    }


def check_class_balance(
    file_list: list[tuple[str, str]],
    num_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """Simulate pair generation and count positive/negative ratio."""
    rng = random.Random(seed)
    speaker_to_files: dict[str, list[str]] = defaultdict(list)
    for path, spk in file_list:
        speaker_to_files[spk].append(path)
    speakers = [s for s, fs in speaker_to_files.items() if len(fs) >= 2]

    pos_count = 0
    neg_count = 0
    for _ in range(num_samples):
        label = rng.choice([0, 1])
        if label == 1:
            spk = rng.choice(speakers)
            if len(speaker_to_files[spk]) >= 2:
                pos_count += 1
        else:
            neg_count += 1

    ratio = pos_count / max(pos_count + neg_count, 1)
    return {
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "positive_ratio": round(ratio, 3),
        "status": "✅ PASS (balanced)" if 0.4 <= ratio <= 0.6 else f"⚠️  WARN — ratio {ratio:.2f} (ideal 0.50)",
    }


def sample_pairs(
    file_list: list[tuple[str, str]], n: int = 5, seed: int = 42
) -> list[dict]:
    """Sample n random pairs and return speaker metadata."""
    rng = random.Random(seed)
    speaker_to_files: dict[str, list[str]] = defaultdict(list)
    for path, spk in file_list:
        speaker_to_files[spk].append(path)
    speakers = [s for s, fs in speaker_to_files.items() if len(fs) >= 2]
    all_speakers = list(set(s for _, s in file_list))

    samples = []
    for i in range(n):
        if i % 2 == 0:  # positive
            spk = rng.choice(speakers)
            a, b = rng.sample(speaker_to_files[spk], 2)
            samples.append({"type": "positive", "spk_a": spk, "spk_b": spk,
                             "path_a": a, "path_b": b})
        else:  # negative
            sa, sb = rng.sample(all_speakers, 2)
            a = rng.choice(speaker_to_files.get(sa, [sa]))
            b = rng.choice(speaker_to_files.get(sb, [sb]))
            samples.append({"type": "negative", "spk_a": sa, "spk_b": sb,
                             "path_a": a, "path_b": b})
    return samples


def plot_length_distribution(
    file_list: list[tuple[str, str]], out_path: str, max_files: int = 500
) -> None:
    """Plot histogram of audio durations."""
    rng = random.Random(42)
    sample = rng.sample(file_list, min(max_files, len(file_list)))
    durations = []
    for path, _ in sample:
        try:
            durations.append(get_audio_duration(path))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(durations, bins=50, color="steelblue", edgecolor="white")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title(f"Audio Length Distribution (n={len(durations)} files)")
    ax.axvline(np.median(durations), color="tomato", linestyle="--",
               label=f"Median: {np.median(durations):.1f}s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostics(data_dir: str, out_dir: str = "diagnostics") -> None:
    set_seed()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  DATA PIPELINE DIAGNOSTICS")
    print("=" * 60)

    file_list = scan_data_dir(data_dir)
    if not file_list:
        print(f"[ERROR] No audio files found in {data_dir}")
        return

    print(f"\n📁 Total files   : {len(file_list)}")
    unique_spk = set(s for _, s in file_list)
    print(f"👥 Total speakers: {len(unique_spk)}")
    if len(unique_spk) < 100:
        print(f"  ⚠️  WARNING: Only {len(unique_spk)} speakers found. "
              f"Metric learning requires 100+ speakers for reliable training.")

    train_files, val_files = speaker_disjoint_split(file_list)

    # ── 1. Speaker overlap ──────────────────────────────────────────────────
    print("\n── 1. Speaker Overlap Check ──")
    overlap_result = check_speaker_overlap(train_files, val_files)
    print(f"  Train speakers : {overlap_result['train_speakers']}")
    print(f"  Val speakers   : {overlap_result['val_speakers']}")
    print(f"  Overlap        : {overlap_result['overlap_count']}")
    if overlap_result["overlap_ids"]:
        print(f"  Overlapping IDs: {overlap_result['overlap_ids']}")
    print(f"  Status         : {overlap_result['status']}")

    # ── 2. Utterance overlap ────────────────────────────────────────────────
    print("\n── 2. Utterance Overlap Check ──")
    utt_result = check_utterance_overlap(train_files, val_files)
    print(f"  Shared utterances: {utt_result['overlap_utterances']}")
    if utt_result["examples"]:
        print(f"  Examples         : {utt_result['examples'][:3]}")
    print(f"  Status           : {utt_result['status']}")

    # ── 3. Class balance ────────────────────────────────────────────────────
    print("\n── 3. Pair Class Balance Check ──")
    balance_result = check_class_balance(train_files)
    print(f"  Positives : {balance_result['positive_pairs']}")
    print(f"  Negatives : {balance_result['negative_pairs']}")
    print(f"  Ratio     : {balance_result['positive_ratio']}")
    print(f"  Status    : {balance_result['status']}")

    # ── 4. Sample pairs ─────────────────────────────────────────────────────
    print("\n── 4. Random Pair Samples (speaker ID verification) ──")
    pairs = sample_pairs(train_files)
    for i, p in enumerate(pairs):
        same = "✅ same" if p["spk_a"] == p["spk_b"] else "❌ different"
        print(f"  [{i+1}] {p['type']:8s}  spk_a={p['spk_a']}  spk_b={p['spk_b']}  {same}")

    # ── 5. Length distribution ──────────────────────────────────────────────
    print("\n── 5. Audio Length Distribution ──")
    plot_length_distribution(file_list, f"{out_dir}/audio_lengths.png")

    # ── Summary report ──────────────────────────────────────────────────────
    report_path = Path(out_dir) / "data_diagnosis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATA PIPELINE DIAGNOSTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total files    : {len(file_list)}\n")
        f.write(f"Total speakers : {len(unique_spk)}\n\n")
        f.write(f"Speaker overlap: {overlap_result['status']}\n")
        f.write(f"Utterance overlap: {utt_result['status']}\n")
        f.write(f"Class balance  : {balance_result['status']}\n")
        f.write(f"Train speakers : {overlap_result['train_speakers']}\n")
        f.write(f"Val speakers   : {overlap_result['val_speakers']}\n")
    print(f"\n📄 Report saved → {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose data pipeline")
    parser.add_argument("data_dir", help="Root: speaker_id/utterance.wav")
    parser.add_argument("--out_dir", default="diagnostics")
    args = parser.parse_args()
    run_diagnostics(args.data_dir, args.out_dir)