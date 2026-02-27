import os
import random
import torch
import numpy as np
from sklearn.metrics import roc_curve

from speaker_pairup import SpeakerPairUpDataset
from model import SiameseNetwork


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/siamese_final_epoch_10.pth"

model = SiameseNetwork().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Model loaded for evaluation")


def generate_eval_pairs(dataset, num_pairs=1000):
    #generate new pairs for evaluation coz evalutation should no be done on training pairs itself.
    pairs = []

    for _ in range(num_pairs):
        is_positive = random.random() < 0.5

        if is_positive:
            speaker = random.choice(dataset.speakers)
            file1, file2 = random.sample(dataset.speaker_to_files[speaker], 2)
            label = 1
        else:
            speaker1, speaker2 = random.sample(dataset.speakers, 2)
            file1 = random.choice(dataset.speaker_to_files[speaker1])
            file2 = random.choice(dataset.speaker_to_files[speaker2])
            label = 0

        spec1 = torch.tensor(np.load(file1), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spec2 = torch.tensor(np.load(file2), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        spec1 = dataset.pad_or_crop(spec1)
        spec2 = dataset.pad_or_crop(spec2)

        pairs.append((spec1, spec2, label))

    return pairs

def compute_similarity(model, pairs, device):
    scores = []
    labels = []

    model.eval()
    with torch.no_grad():
        for spec1, spec2, label in pairs:
            spec1 = spec1.to(device)
            spec2 = spec2.to(device)

            emb1, emb2 = model(spec1, spec2)
            sim = torch.nn.functional.cosine_similarity(emb1, emb2)

            scores.append(sim.item())
            labels.append(label)

    return np.array(scores), np.array(labels)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)

    fnr = 1 - tpr
    eer_index = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_index]

    return eer, thresholds[eer_index]


FEATURES_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/features"

dataset = SpeakerPairUpDataset(FEATURES_DATA_DIR)

print("Generating evaluation pairs...")
pairs = generate_eval_pairs(dataset, num_pairs=1000)

print("Computing similarity scores...")
scores, labels = compute_similarity(model, pairs, DEVICE)

eer, threshold = compute_eer(scores, labels)

print("\n===== EVALUATION RESULTS =====")
print(f"EER (Equal Error Rate): {eer * 100:.2f}%")
print(f"Optimal Threshold at EER: {threshold:.4f}")
