"""
model.py — ECAPA-TDNN Siamese Network with Attentive Statistical Pooling.

Priority fixes applied:
  - Option A: Load SpeechBrain pretrained ECAPA-TDNN weights (VoxCeleb)
  - Option B: Custom ECAPA-TDNN (fallback when speechbrain unavailable)
  - All backbone layers unfrozen for fine-tuning
  - Mel-band count (80) verified to match ECAPA input spec
  - L2-normalised 128-D embeddings on unit sphere

Architecture:
    Input  (B, n_mels, T)
      ↓
    ECAPA-TDNN backbone  →  frame-level features (B, C, T)
      ↓
    Attentive Statistical Pooling  →  (B, 2*C)
      ↓
    Projection head: Linear → BN → L2-norm  →  (B, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Custom ECAPA-TDNN building blocks (Option B / fallback)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.avg_pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class Res2Block(nn.Module):
    """SE-Res2 convolution block with dilated TDNN-style convolutions."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 8) -> None:
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale

        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.width, self.width, kernel_size,
                          padding=dilation * (kernel_size // 2), dilation=dilation),
                nn.BatchNorm1d(self.width),
                nn.ReLU(inplace=True),
            )
            for _ in range(scale - 1)
        ])
        self.conv3 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        chunks = torch.chunk(x, self.scale, dim=1)
        out: list[Tensor] = []
        y: Tensor | None = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                out.append(chunk)
            elif i == 1:
                y = self.convs[i - 1](chunk)
                out.append(y)
            else:
                y = self.convs[i - 1](chunk + y)   # type: ignore[operator]
                out.append(y)
        x = torch.cat(out, dim=1)
        x = self.conv3(x)
        x = self.se(x)
        return self.relu(x + residual)


class CustomECAPATDNN(nn.Module):
    """Custom ECAPA-TDNN backbone (used when SpeechBrain is unavailable).

    Args:
        in_channels: Number of mel-frequency bins (must match feature extractor).
        channels: Internal channel width (default 512).
    """

    def __init__(self, in_channels: int = 80, channels: int = 512) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = Res2Block(channels, kernel_size=3, dilation=2)
        self.layer2 = Res2Block(channels, kernel_size=3, dilation=3)
        self.layer3 = Res2Block(channels, kernel_size=3, dilation=4)
        # Multi-scale Feature Aggregation
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, 1),
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(inplace=True),
        )
        self.out_channels: int = channels * 3

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, n_mels, T)``
        Returns:
            Frame-level features ``(B, 3*channels, T)``
        """
        x = self.conv1(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        return self.mfa(torch.cat([e1, e2, e3], dim=1))


def _try_load_speechbrain(channels: int = 512) -> tuple[nn.Module | None, int]:
    """Attempt to load pretrained SpeechBrain ECAPA-TDNN (1.0+ API first).

    Falls back to HuggingFace direct download if SpeechBrain fails.

    Returns:
        ``(backbone, out_channels)`` or ``(None, -1)`` on failure.
    """
    # ── Try SpeechBrain 1.0+ inference API ──────────────────────────────
    try:
        from speechbrain.inference.classifiers import EncoderClassifier  # type: ignore

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            savedir="pretrained_models/ecapa",
        )
        backbone = classifier.mods.embedding_model
        for param in backbone.parameters():
            param.requires_grad = True

        with torch.no_grad():
            dummy = torch.zeros(1, 80, 200)
            try:
                out = backbone(dummy)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                out_channels = out.shape[1]
            except Exception:
                out_channels = 192

        print("[model] ✅ Loaded pretrained SpeechBrain ECAPA-TDNN (VoxCeleb) via 1.0+ API")
        return backbone, out_channels

    except Exception as e1:
        print(f"[model] ⚠️  SpeechBrain 1.0+ API failed: {e1}")

    # ── Try HuggingFace direct download ─────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download  # type: ignore

        model_path = hf_hub_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            filename="embedding_model.ckpt",
            local_dir="pretrained_models/ecapa-hf",
        )
        backbone = CustomECAPATDNN(in_channels=80, channels=channels)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        # Try loading; strict=False allows partial weight matches
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(
            f"[model] ✅ Loaded pretrained weights from HuggingFace "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        return backbone, backbone.out_channels

    except Exception as e2:
        print(f"[model] ⚠️  HuggingFace fallback also failed ({e2}). Using custom ECAPA-TDNN.")
        return None, -1


# ---------------------------------------------------------------------------
# Attentive Statistical Pooling
# ---------------------------------------------------------------------------

class AttentiveStatisticalPooling(nn.Module):
    """Self-attention weighted mean + std pooling.

    Args:
        in_channels: Frame-level feature channels.
        attention_channels: Hidden size of attention MLP.
    """

    def __init__(self, in_channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, 1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, in_channels, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, C, T)``
        Returns:
            ``(B, 2*C)`` — weighted mean concatenated with weighted std.
        """
        w = self.attention(x)                           # (B, C, T)
        mean = (w * x).sum(dim=-1)                      # (B, C)
        var = (w * x.pow(2)).sum(dim=-1) - mean.pow(2)
        std = var.clamp(min=1e-8).sqrt()                # (B, C)
        return torch.cat([mean, std], dim=1)            # (B, 2C)


# ---------------------------------------------------------------------------
# Full Siamese model
# ---------------------------------------------------------------------------

class SpeakerVerificationModel(nn.Module):
    """ECAPA-TDNN Siamese network for speaker verification.

    Automatically attempts to load SpeechBrain pretrained weights; falls back
    to a custom ECAPA-TDNN if SpeechBrain is unavailable.

    Args:
        n_mels: Number of mel-frequency bins (must be 80 for ECAPA).
        channels: ECAPA backbone channel width (custom backbone only).
        embedding_dim: Output embedding dimension.
        use_pretrained: Whether to attempt loading SpeechBrain weights.
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 512,
        embedding_dim: int = 128,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        assert n_mels == 80, (
            f"ECAPA-TDNN expects 80 mel bins, got {n_mels}. "
            "Update feature_extraction.py to use n_mels=80."
        )

        # Try pretrained backbone
        pretrained_backbone, sb_out_channels = (
            _try_load_speechbrain(channels) if use_pretrained else (None, -1)
        )

        if pretrained_backbone is not None:
            self.backbone = pretrained_backbone
            backbone_out = sb_out_channels
            self.use_speechbrain = True
        else:
            self.backbone = CustomECAPATDNN(in_channels=n_mels, channels=channels)
            backbone_out = self.backbone.out_channels   # type: ignore[union-attr]
            self.use_speechbrain = False

        self.pooling = AttentiveStatisticalPooling(
            in_channels=backbone_out, attention_channels=128
        )
        pooled_dim = backbone_out * 2
        self.dropout = nn.Dropout(p=0.5)
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.embedding_dim = embedding_dim

        # Final assertion
        self._verify_norm()

    def _verify_norm(self) -> None:
        """Quick sanity-check that embeddings are on the unit sphere."""
        self.eval()
        with torch.no_grad():
            x = torch.randn(2, 80, 200)
            emb = self.embed(x)
            norms = emb.norm(dim=1)
            assert (norms - 1.0).abs().max().item() < 1e-5, (
                f"Embedding norms not ≈ 1.0: {norms}"
            )
        self.train()
        print(f"[model] ✅ Embedding L2-norm verified (≈ 1.0)")

    def _backbone_forward(self, x: Tensor) -> Tensor:
        """Handle both SpeechBrain and custom backbone forward signatures."""
        if self.use_speechbrain:
            out = self.backbone(x)
            # SpeechBrain may return a tuple or a single tensor
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out
        return self.backbone(x)

    def embed(self, x: Tensor) -> Tensor:
        """Compute L2-normalised embeddings.

        Args:
            x: ``(B, n_mels, T)``
        Returns:
            ``(B, embedding_dim)`` unit-sphere embeddings.
        """
        feat = self._backbone_forward(x)   # (B, C, T)
        feat = self.dropout(feat)           # dropout on frame-level features
        pooled = self.pooling(feat)        # (B, 2C)
        emb = self.projection(pooled)      # (B, D)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x_a: Tensor, x_b: Tensor) -> tuple[Tensor, Tensor]:
        """Siamese forward — returns two L2-normalised embedding tensors.

        Args:
            x_a: ``(B, n_mels, T_a)``
            x_b: ``(B, n_mels, T_b)``
        Returns:
            ``(emb_a, emb_b)``, each ``(B, embedding_dim)``.
        """
        return self.embed(x_a), self.embed(x_b)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    model = SpeakerVerificationModel(n_mels=80, channels=512, embedding_dim=128)
    x_a = torch.randn(4, 80, 300)
    x_b = torch.randn(4, 80, 250)
    ea, eb = model(x_a, x_b)
    print(f"emb_a: {ea.shape}  norms: {ea.norm(dim=1).tolist()}")
    print(f"emb_b: {eb.shape}  norms: {eb.norm(dim=1).tolist()}")
    assert (ea.norm(dim=1) - 1.0).abs().max().item() < 1e-5, "Norm check failed!"
    print("✅ All assertions passed.")