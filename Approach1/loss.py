import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # emb1, emb2: (B, 128)
        # label: (B,)  -> 1 same, 0 different

        # Cosine distance
        cosine_sim = F.cosine_similarity(emb1, emb2)
        distance = 1.0 - cosine_sim

        positive_loss = label * torch.pow(distance, 2)
        negative_loss = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )

        loss = torch.mean(positive_loss + negative_loss)
        return loss
