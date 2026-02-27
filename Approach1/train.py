# import os
# import torch
# from torch.utils.data import DataLoader
# from speaker_pairup import SpeakerPairUpDataset
# from model import SiameseNetwork
# from loss import ContrastiveLoss

# os.makedirs("checkpoints", exist_ok=True)

# FEATURES_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/features"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = SpeakerPairUpDataset(FEATURES_DATA_DIR)
# loader = DataLoader(
#     dataset,
#     batch_size=16,
#     shuffle=True,
#     num_workers=6,
#     pin_memory=True
# )

# model = SiameseNetwork().to(DEVICE)
# criterion = ContrastiveLoss(margin=1.0)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# EPOCHS = 10
# best_loss = float("inf")

# for epoch in range(EPOCHS):
#     model.train()
#     running_loss = 0.0

#     print(f"Starting epoch {epoch+1}/{EPOCHS}")
#     for spec1, spec2, label in loader:
#         spec1 = spec1.to(DEVICE)
#         spec2 = spec2.to(DEVICE)
#         label = label.to(DEVICE)

#         optimizer.zero_grad()

#         emb1, emb2 = model(spec1, spec2)
#         loss = criterion(emb1, emb2, label)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(loader)
#     print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")


# torch.save({
#         "epoch": EPOCHS,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#     }, f"checkpoints/speaker_verification_epoch_{EPOCHS}.pth")



import os
import torch
from torch.utils.data import DataLoader

from speaker_pairup import SpeakerPairUpDataset
from model import SiameseNetwork
from loss import ContrastiveLoss

# =========================
# Configuration
# =========================
FEATURES_DATA_DIR = "/home/rohithkaki/Voice_Biometrics/data/features"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
MARGIN = 1.0
NUM_WORKERS = 6   # increase to 4 if CPU allows

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Setup
# =========================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Number of DataLoader workers: {NUM_WORKERS}")

# =========================
# Dataset & DataLoader
# =========================
dataset = SpeakerPairUpDataset(FEATURES_DATA_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

# =========================
# Model, Loss, Optimizer
# =========================
model = SiameseNetwork().to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Model, loss, and optimizer initialized.")
print(f"Training for {EPOCHS} epochs...\n")

# =========================
# Training Loop
# =========================
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    print(f"========== Epoch {epoch + 1}/{EPOCHS} ==========")

    for batch_idx, (spec1, spec2, label) in enumerate(loader):
        spec1 = spec1.to(DEVICE, non_blocking=True)
        spec2 = spec2.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        emb1, emb2 = model(spec1, spec2)
        loss = criterion(emb1, emb2, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(
                f"Batch [{batch_idx + 1}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(loader)
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

    # =========================
    # Save BEST model
    # =========================
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_path = os.path.join(
            CHECKPOINT_DIR, "best_siamese_model.pth"
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            },
            best_model_path
        )
        print(f"✅ Best model saved (loss = {best_loss:.4f})")

    print("--------------------------------------------\n")

# =========================
# Save FINAL model
# =========================
final_model_path = os.path.join(
    CHECKPOINT_DIR, f"siamese_final_epoch_{EPOCHS}.pth"
)

torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    final_model_path
)

print("🎉 Training complete.")
print(f"Final model saved at: {final_model_path}")
print(f"Best model saved at: {best_model_path}")
