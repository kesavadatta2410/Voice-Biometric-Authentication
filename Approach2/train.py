# import torch
# from torch.utils.data import DataLoader
# from dataset import SpeakerPairDataset
# from model import SiameseNet
# from loss import ContrastiveLoss

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dataset = SpeakerPairDataset("/home/rohithkaki/Voice_Biometrics/data/features")
# loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

# model = SiameseNet().to(DEVICE)
# criterion = ContrastiveLoss()
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=1e-4
# )

# for epoch in range(10):
#     model.train()
#     total = 0
#     for x1, x2, y in loader:
#         x1, x2, y = x1.to(DEVICE), x2.to(DEVICE), y.to(DEVICE)
#         e1, e2 = model(x1, x2)
#         loss = criterion(e1, e2, y)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total += loss.item()

#     print(f"Epoch {epoch+1}, Loss: {total/len(loader):.4f}")

# torch.save(model.state_dict(), "siamese_vggish.pth")



import torch
from torch.utils.data import DataLoader
from dataset_pairup import SpeakerPairDataset
from model import SiameseNet
from loss import ContrastiveLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    dataset = SpeakerPairDataset("/home/rohithkaki/Voice_Biometrics/data/features_mels_64")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)

    model = SiameseNet().to(DEVICE)
    criterion = ContrastiveLoss()
    
    # Only optimize parameters that have requires_grad=True (the projection layers)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    for epoch in range(30):
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

        print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "siamese_vggish_20_epochs.pth")

if __name__ == "__main__":
    train()