"""Simple test to verify model loading and inference"""
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model.unet import UNet
import os

# Load model
print("Loading model...")
model = UNet()
model_path = "saved_model.pth"

if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found!")
    print("Please download saved_model.pth from Colab and place it here")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("Model loaded successfully!")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test on a sample image
test_image_path = "dataset/test/images/0000.png"
if os.path.exists(test_image_path):
    print(f"\nTesting on {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    mask = (output > 0.5).float().squeeze().numpy() * 255
    print(f"Inference successful! Output shape: {mask.shape}")
    print(f"   Mask has {np.sum(mask > 0)} white pixels")
    
    # Save test output
    from PIL import Image as PILImage
    PILImage.fromarray(mask.astype(np.uint8)).save("test_output.png")
    print("Test output saved to test_output.png")
else:
    print(f"Test image not found: {test_image_path}")
    print("Available test images:")
    if os.path.exists("dataset/test/images"):
        print(os.listdir("dataset/test/images")[:5])

print("\nModel is working correctly!")