import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # x shape: (batch_size, 1, 80, T)
        x = F.relu(self.conv1(x))
        x = self.pool(x) #(batch_size, 32, 40, T/2)

        x = F.relu(self.conv2(x))
        x = self.pool(x) #(batch_size, 64, 20, T/4)  Each pooling: halves frequency, halves time
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
    def forward(self, x):
        # x: (B, T, feature_dim)
        outputs, hidden = self.gru(x)

        # hidden: (1, B, hidden_dim)
        return hidden.squeeze(0)
    

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
                # Why L2 normalization?
                # Embeddings lie on unit hypersphere
                # Makes cosine similarity meaningful
                # Standard in speaker verification
    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SpeakerEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = CNNEncoder()
        self.temporal = TemporalEncoder(input_dim=64 * 20)
        self.projector = ProjectionHead(input_dim=128)

    def forward(self, x):
        x = self.cnn(x)

        # CNN output (B, channels, freq, time)
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, C * F)  # (B, T, C*F)
        #RNN input shape (B, sequence_length, feature_dim)

        x = self.temporal(x)
        x = self.projector(x)

        return x


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = SpeakerEmbeddingNet()

    def forward(self, x1, x2):
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        return emb1, emb2


