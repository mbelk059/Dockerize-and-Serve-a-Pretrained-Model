import os, torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from model.unet import UNet
from dataset import get_dataloaders

load_dotenv()

def compute_iou(pred, target):
    pred   = (pred > 0.5).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter / union).item() if union > 0 else 1.0

def compute_dice(pred, target):
    pred  = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum())).item() if (pred.sum() + target.sum()) > 0 else 1.0

def evaluate():
    model_path = os.getenv("MODEL_PATH", "./saved_model.pth")
    data_dir   = os.getenv("DATA_DIR", "./dataset")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = get_dataloaders(data_dir, batch_size=1)
    model   = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ious, dices = [], []
    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loaders["test"]):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            ious.append(compute_iou(preds[0], masks[0]))
            dices.append(compute_dice(preds[0], masks[0]))

            # Save visualization
            pred_np = (preds[0,0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_np).save(f"predictions/pred_{i:04d}.png")

    print(f"Test IoU:  {np.mean(ious):.4f}")
    print(f"Test Dice: {np.mean(dices):.4f}")

if __name__ == "__main__":
    evaluate()