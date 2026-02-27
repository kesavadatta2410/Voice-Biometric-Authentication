"""
preprocess.py — Audio preprocessing pipeline for ECAPA-TDNN speaker verification.

Steps:
1. Resample to 16 kHz
2. Trim silence via energy-based VAD
3. Amplitude normalisation
"""

import random
import numpy as np
import torch
import torch.nn.functional as torch_F
import torchaudio.functional as F
import soundfile as sf
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


TARGET_SR: int = 16_000


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load a waveform from *path* and return ``(waveform, sample_rate)``.

    Uses soundfile for reading to avoid torchaudio backend issues on Windows.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, channels)
    waveform = torch.from_numpy(data.T)  # (channels, T)
    return waveform, sr


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Resample *waveform* from *orig_sr* to *target_sr*."""
    if orig_sr == target_sr:
        return waveform
    return F.resample(waveform, orig_sr, target_sr)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel audio to mono by averaging channels."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def energy_vad_trim(
    waveform: torch.Tensor,
    sr: int = TARGET_SR,
    frame_ms: int = 20,
    energy_threshold_db: float = -50.0,
    min_speech_frames: int = 5,
) -> torch.Tensor:
    """Trim leading/trailing silence using an energy-based VAD.

    Args:
        waveform: Shape ``(1, T)``.
        sr: Sample rate in Hz.
        frame_ms: Frame length in milliseconds.
        energy_threshold_db: dB threshold below which a frame is silent.
        min_speech_frames: Minimum number of consecutive speech frames to keep.

    Returns:
        Trimmed waveform ``(1, T')``.
    """
    frame_len = int(sr * frame_ms / 1000)
    signal = waveform.squeeze(0)  # (T,)

    # Pad so length is a multiple of frame_len
    remainder = len(signal) % frame_len
    if remainder:
        signal = torch.nn.functional.pad(signal, (0, frame_len - remainder))

    frames = signal.reshape(-1, frame_len)  # (N, frame_len)
    energy_db = 10 * torch.log10(frames.pow(2).mean(dim=1) + 1e-10)
    speech_mask = energy_db > energy_threshold_db  # (N,)

    speech_indices = speech_mask.nonzero(as_tuple=False).flatten()
    if len(speech_indices) < min_speech_frames:
        # No reliable speech detected — return as-is
        return waveform

    start_frame = int(speech_indices[0].item())
    end_frame = int(speech_indices[-1].item()) + 1

    start_sample = start_frame * frame_len
    end_sample = end_frame * frame_len
    trimmed = signal[start_sample:end_sample].unsqueeze(0)
    return trimmed


def amplitude_normalise(waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise waveform amplitude to [-1, 1] peak."""
    peak = waveform.abs().max()
    if peak > eps:
        waveform = waveform / peak
    return waveform


def preprocess(path: str) -> torch.Tensor:
    """Full preprocessing pipeline.

    Args:
        path: Path to an audio file.

    Returns:
        Preprocessed mono waveform tensor of shape ``(1, T)`` at 16 kHz.
    """
    waveform, sr = load_audio(path)
    waveform = to_mono(waveform)
    waveform = resample(waveform, sr)
    waveform = energy_vad_trim(waveform)
    waveform = amplitude_normalise(waveform)
    return waveform


if __name__ == "__main__":
    set_seed()
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <audio_file>")
        sys.exit(1)

    wav = preprocess(sys.argv[1])
    print(f"Preprocessed waveform shape: {wav.shape}, SR: {TARGET_SR} Hz")