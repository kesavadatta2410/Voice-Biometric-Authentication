import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGishEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the VGGish backbone
        base_vggish = torch.hub.load("harritaylor/torchvggish", "vggish")
        
        # We only take the 'features' part (the convolutional layers)
        # This allows us to handle variable lengths like your 400 frames
        self.features = base_vggish.features
        
        # Freeze the pretrained weights
        for p in self.features.parameters():
            p.requires_grad = False  

        # VGGish features output 512 channels. We pool them and project to 128.
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) 
        self.proj = nn.Linear(512, 128) # Project 512-D features to 128-D embedding

    def forward(self, x):
        # x shape: (B, 1, 64, 400)
        
        # 1. Extract convolutional features
        x = self.features(x) # Output: (B, 512, H_feat, W_feat)
        
        # 2. Global Average Pool: reduces any H/W to 1x1
        x = self.pooling(x) # Output: (B, 512, 1, 1)
        x = x.view(x.size(0), -1) # Flatten: (B, 512)
        
        # 3. Project and Normalize
        emb = self.proj(x)
        return F.normalize(emb, p=2, dim=1)

# This MUST be in the same file to fix your ImportError
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VGGishEncoder()

    def forward(self, x1, x2):
        # Generate embeddings for the pair
        return self.encoder(x1), self.encoder(x2)