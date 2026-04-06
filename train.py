import os, torch, torch.nn as nn
from torch.optim import Adam
from dotenv import load_dotenv
from model.unet import UNet
from dataset import get_dataloaders

load_dotenv()

def dice_loss(pred, target, smooth=1):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train():
    model_path = os.getenv("MODEL_PATH", "./saved_model.pth")
    data_dir   = os.getenv("DATA_DIR", "./dataset")
    epochs     = int(os.getenv("EPOCHS", 20))
    batch_size = int(os.getenv("BATCH_SIZE", 4))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(data_dir, batch_size=batch_size)
    model   = UNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0
        for imgs, masks in loaders["train"]:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(loaders["train"])

        # --- val ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in loaders["val"]:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                val_loss += dice_loss(preds, masks).item()
        val_loss /= len(loaders["val"])

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to {model_path}")

if __name__ == "__main__":
    train()