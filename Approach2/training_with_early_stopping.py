import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_pairup import SpeakerPairDataset
from model import SiameseNet
from loss import ContrastiveLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EarlyStopping:
    """Stops training if loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0, path='siamese_vggish_best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when loss decreases.'''
        torch.save(model.state_dict(), self.path)
        print(f"Checkpoint saved: Loss improved to {self.best_loss:.4f}")

def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Loss curve saved as training_loss_curve.png")

def train():
    # 1. Setup Data and Model
    dataset = SpeakerPairDataset("/home/rohithkaki/Voice_Biometrics/data/features_mels_64")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

    model = SiameseNet().to(DEVICE)
    criterion = ContrastiveLoss()
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # 2. Initialize Early Stopping and History
    early_stopping = EarlyStopping(patience=5, path="siamese_vggish_best.pth")
    loss_history = []

    print(f"Starting training on {DEVICE}...")

    # Set a high range; EarlyStopping will handle the exit
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        
        for x1, x2, y in loader:
            x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
            
            e1, e2 = model(x1, x2)
            loss = criterion(e1, e2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        # 3. Check Early Stopping
        early_stopping(avg_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

    # 4. Final Visuals
    plot_loss(loss_history)
    # Also save the very last state just in case
    torch.save(model.state_dict(), "siamese_vggish_final_state.pth")

if __name__ == "__main__":
    train()