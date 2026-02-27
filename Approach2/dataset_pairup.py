import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpeakerPairDataset(Dataset):
    MAX_TIME = 400

    def __init__(self, feature_dir):
        self.speaker_to_files = {}
        for spk in os.listdir(feature_dir):
            spk_path = os.path.join(feature_dir, spk)
            if os.path.isdir(spk_path):
                self.speaker_to_files[spk] = [
                    os.path.join(spk_path, f)
                    for f in os.listdir(spk_path)
                    if f.endswith(".npy")
                ]
        self.speakers = list(self.speaker_to_files.keys())

    def pad_or_crop(self, spec):
        T = spec.shape[-1]
        if T > self.MAX_TIME:
            return spec[..., :self.MAX_TIME]
        elif T < self.MAX_TIME:
            return F.pad(spec, (0, self.MAX_TIME - T))
        return spec

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        same = random.random() < 0.5

        if same:
            spk = random.choice(self.speakers)
            f1, f2 = random.sample(self.speaker_to_files[spk], 2)
            label = 1
        else:
            spk1, spk2 = random.sample(self.speakers, 2)
            f1 = random.choice(self.speaker_to_files[spk1])
            f2 = random.choice(self.speaker_to_files[spk2])
            label = 0

        s1 = torch.tensor(np.load(f1)).unsqueeze(0)
        s2 = torch.tensor(np.load(f2)).unsqueeze(0)

        s1 = self.pad_or_crop(s1)
        s2 = self.pad_or_crop(s2)

        return s1, s2, torch.tensor(label, dtype=torch.float32)
