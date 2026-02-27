import os
import random
import numpy as np
import torch 
from torch.utils.data import Dataset

class SpeakerPairUpDataset(Dataset):
    MAX_TIME_FRAMES = 400

    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.speaker_to_files = {}
        for speaker in os.listdir(feature_dir):
            speaker_path = os.path.join(feature_dir, speaker)
            if os.path.isdir(speaker_path):
                self.speaker_to_files[speaker] = [os.path.join(speaker_path, f) for f in os.listdir(speaker_path) if f.endswith('.npy')]
        self.speakers = list(self.speaker_to_files.keys())
    
    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        # Decide positive or negative pair
        is_positive = random.random() < 0.5
        if is_positive:
            # pair from same speaker with different emotion files
            speaker = random.choice(self.speakers)
            file1, file2 = random.sample(self.speaker_to_files[speaker], 2)
            label = 1
        else:
            #pair form different speakers with different emotion files
            speaker1, speaker2 = random.sample(self.speakers, 2)
            file1 = random.choice(self.speaker_to_files[speaker1])
            file2 = random.choice(self.speaker_to_files[speaker2])
            label = 0

        #add channel dimension CNN expects (channels, height, width) using unsqueeze (Use this when you specifically want to add a dimension of size 1.)   
        # (1, 80, T): 1 → channel, 80 → Mel frequency bins, T → time frames #
        # (spec = log_mel_spectrogram) converting from numpy array to torch tensor
        spec1 = torch.tensor(np.load(file1), dtype=torch.float32).unsqueeze(0) 
        spec2 = torch.tensor(np.load(file2), dtype=torch.float32).unsqueeze(0)
        spec1 = self.pad_or_crop(spec1)
        spec2 = self.pad_or_crop(spec2)
        return spec1, spec2, torch.tensor(label, dtype=torch.float32)
    

    def pad_or_crop(self, spec, max_len=MAX_TIME_FRAMES):
        """
        Supports spec shapes:
        - (1, 80, T)
        - (1, 1, 80, T)
        """

        if spec.dim() == 3:
            # (C, 80, T)
            _, _, T = spec.shape
            batch_dim = False
        elif spec.dim() == 4:
            # (B, C, 80, T)
            _, _, _, T = spec.shape
            batch_dim = True
        else:
            raise ValueError(f"Unexpected spec shape: {spec.shape}")

        if T > max_len:
            spec = spec[..., :max_len]
        elif T < max_len:
            pad_amount = max_len - T
            spec = torch.nn.functional.pad(
                spec,
                (0, pad_amount),
                mode="constant",
                value=0.0
            )
        return spec
