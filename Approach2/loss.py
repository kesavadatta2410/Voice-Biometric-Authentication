import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, y):
        d = 1 - F.cosine_similarity(e1, e2)
        loss = y * d.pow(2) + (1 - y) * torch.clamp(self.margin - d, min=0).pow(2)
        return loss.mean()
