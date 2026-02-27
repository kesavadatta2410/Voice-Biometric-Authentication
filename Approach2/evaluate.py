import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from dataset_pairup import SpeakerPairDataset
from model import SiameseNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation():
    # 1. Setup
    model = SiameseNet().to(DEVICE)
    model.load_state_dict(torch.load("siamese_vggish_final_state.pth", map_location=DEVICE))
    model.eval()

    # 2. Data Loading (Ensure this is a VALIDATION/TEST set if possible)
    dataset = SpeakerPairDataset("/home/rohithkaki/Voice_Biometrics/data/features_mels_64")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    scores, labels = [], []

    print("Evaluating Model Performance...")
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(loader):
            if i > 100: break # Checking ~6400 pairs for better statistics
            
            e1, e2 = model(x1.to(DEVICE), x2.to(DEVICE))
            
            # Compute similarity (range -1 to 1)
            sim = torch.cosine_similarity(e1, e2)
            scores.extend(sim.cpu().numpy())
            labels.extend(y.numpy())

    scores = np.array(scores)
    labels = np.array(labels)

    # 3. Calculate EER
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[idx]
    optimal_threshold = thresholds[idx]

    print("-" * 30)
    print(f"Equal Error Rate (EER): {eer*100:.2f}%")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print("-" * 30)

    # 4. Visualize the Results
    plot_distributions(scores, labels, optimal_threshold)

def plot_distributions(scores, labels, threshold):
    same_speaker = scores[labels == 1]
    diff_speaker = scores[labels == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(same_speaker, bins=50, alpha=0.6, label='Same Speaker (Positive)', color='green')
    plt.hist(diff_speaker, bins=50, alpha=0.6, label='Different Speaker (Negative)', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
    
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_evaluation()