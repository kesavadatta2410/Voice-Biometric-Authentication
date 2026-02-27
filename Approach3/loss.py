"""
loss.py — Loss functions for speaker verification training.

Provides:
  - OnlineTripletLoss : primary metric loss with semi-hard negative mining
  - ContrastiveLoss   : auxiliary pair-level loss
  - TripletLoss       : vectorised semi-hard triplet loss (alternative)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OnlineTripletLoss(nn.Module):
    """Triplet loss with semi-hard negative mining.

    Works well with small speaker counts (e.g. 24 speakers) since it
    mines hard triplets within each mini-batch rather than relying on
    a large class-weight matrix.

    Args:
        margin: Triplet margin. Default 0.5.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Semi-hard online triplet mining.

        Args:
            embeddings: L2-normalised ``(B, D)`` tensor.
            labels: Speaker-class indices ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        device = embeddings.device
        distances = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        triplets = []
        n = len(labels)
        idx = torch.arange(n, device=device)  # keep on same device as labels

        for i in range(n):
            positive_mask = (labels == labels[i]) & (idx != i)
            negative_mask = labels != labels[i]

            if positive_mask.sum() == 0 or negative_mask.sum() == 0:
                continue

            # Hardest positive
            d_ap = distances[i][positive_mask].max()

            # Semi-hard negatives: d(a,n) < d(a,p) + margin
            anchor_neg_dists = distances[i][negative_mask]
            semi_hard_mask = anchor_neg_dists < d_ap + self.margin

            if semi_hard_mask.sum() > 0:
                p_idx = torch.where(positive_mask)[0][distances[i][positive_mask].argmax()]
                n_idx = torch.where(negative_mask)[0][anchor_neg_dists[semi_hard_mask].argmin()]
                triplets.append((i, p_idx.item(), n_idx.item()))

        if len(triplets) == 0:
            return embeddings.sum() * 0.0  # differentiable zero

        triplet_losses = [
            F.relu(distances[a, p] - distances[a, n] + self.margin)
            for a, p, n in triplets
        ]
        return torch.stack(triplet_losses).mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for same/different-speaker pairs.

    Args:
        margin: Minimum distance margin for negative pairs. Default 0.5.
    """

    def __init__(self, margin: float = 0.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, emb_a: Tensor, emb_b: Tensor, labels: Tensor) -> Tensor:
        """Compute contrastive loss.

        Args:
            emb_a: L2-normalised embeddings ``(B, D)``.
            emb_b: L2-normalised embeddings ``(B, D)``.
            labels: 1 for same speaker, 0 for different ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        labels = labels.float()
        dist = F.pairwise_distance(emb_a, emb_b, p=2)
        positive_loss = labels * dist.pow(2)
        negative_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)
        return (positive_loss + negative_loss).mean()


class TripletLoss(nn.Module):
    """Vectorised online semi-hard triplet loss (batch-all strategy).

    Args:
        margin: Triplet margin. Default 0.3.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    @staticmethod
    def _pairwise_distances(embeddings: Tensor) -> Tensor:
        dot = torch.mm(embeddings, embeddings.t())
        sq = dot.diagonal().unsqueeze(1)
        return (sq + sq.t() - 2 * dot).clamp(min=0)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        dist2 = self._pairwise_distances(embeddings)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_ne = ~labels_eq
        loss_values: list[Tensor] = []

        for i in range(embeddings.size(0)):
            pos_mask = labels_eq[i].clone()
            pos_mask[i] = False
            if not pos_mask.any():
                continue
            d_ap = dist2[i][pos_mask].max()
            neg_mask = labels_ne[i]
            semi_hard = neg_mask & (dist2[i] > d_ap) & (dist2[i] < d_ap + self.margin)
            if not semi_hard.any():
                semi_hard = neg_mask
            d_an = dist2[i][semi_hard].min()
            loss_values.append(F.relu(d_ap - d_an + self.margin))

        if not loss_values:
            return embeddings.sum() * 0.0
        return torch.stack(loss_values).mean()