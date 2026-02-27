"""
feature_extraction.py — 80-band log-mel spectrogram extraction for ECAPA-TDNN.

Parameters follow the standard ECAPA-TDNN / x-vector recipe:
  n_fft=512, hop_length=160, n_mels=80, sample_rate=16000
"""

import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path

from preprocess import preprocess, TARGET_SR


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LogMelExtractor:
    """Extract log-mel spectrograms from raw waveforms.

    Args:
        sr: Expected sample rate (Hz).
        n_fft: FFT window size.
        hop_length: Hop length in samples.
        n_mels: Number of mel filter banks.
        f_min: Minimum frequency.
        f_max: Maximum frequency (``None`` → sr/2).
        top_db: Maximum dynamic range for amplitude-to-dB conversion.
    """

    def __init__(
        self,
        sr: int = TARGET_SR,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float | None = None,
        top_db: float = 80.0,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.top_db = top_db

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max else sr // 2,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute log-mel spectrogram.

        Args:
            waveform: Shape ``(1, T)``.

        Returns:
            Log-mel spectrogram of shape ``(1, n_mels, frames)``.
        """
        mel = self.mel_transform(waveform)       # (1, n_mels, frames)
        log_mel = self.amplitude_to_db(mel)      # (1, n_mels, frames)
        # Normalise to zero mean, unit variance per utterance
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        return log_mel


def extract_from_file(path: str, extractor: LogMelExtractor | None = None) -> torch.Tensor:
    """Preprocess an audio file and return its log-mel spectrogram.

    Args:
        path: Path to audio file.
        extractor: Optional pre-built :class:`LogMelExtractor`.

    Returns:
        Tensor of shape ``(1, n_mels, frames)``.
    """
    if extractor is None:
        extractor = LogMelExtractor()
    waveform = preprocess(path)
    return extractor(waveform)


def extract_and_save(path: str, out_dir: str, extractor: LogMelExtractor | None = None) -> Path:
    """Extract features from *path* and save as a ``.npy`` file.

    Args:
        path: Input audio file path.
        out_dir: Directory to write the ``.npy`` file.
        extractor: Optional pre-built :class:`LogMelExtractor`.

    Returns:
        Path to the saved ``.npy`` file.
    """
    features = extract_from_file(path, extractor)
    out_path = Path(out_dir) / (Path(path).stem + ".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), features.numpy())
    return out_path


if __name__ == "__main__":
    set_seed()
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <audio_file> [output_dir]")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    saved = extract_and_save(sys.argv[1], out_dir)
    print(f"Saved features to {saved}")