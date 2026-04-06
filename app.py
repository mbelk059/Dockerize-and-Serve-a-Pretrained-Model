import os
import io
import base64
import logging
from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./saved_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Import UNet model
from model.unet import UNet

# Load model
logger.info(f"Loading segmentation model from {MODEL_PATH}...")
logger.info(f"Using device: {DEVICE}")

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
logger.info("Model loaded successfully!")

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def postprocess_mask(mask_tensor):
    """Convert model output to binary mask image"""
    mask = (mask_tensor > 0.5).float()
    mask = mask.squeeze().cpu().numpy() * 255
    return mask.astype(np.uint8)

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "House Segmentation API",
        "model": "UNet (trained on satellite imagery)",
        "device": DEVICE
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint - accepts image upload and returns segmentation mask
    """
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "Missing 'image' file"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read and preprocess image
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess mask
        mask = postprocess_mask(output)
        
        # Convert mask to PNG and return
        mask_img = Image.fromarray(mask, mode='L')
        
        # Save to bytes
        img_io = io.BytesIO()
        mask_img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """
    Alternative endpoint - accepts base64 image and returns base64 mask
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(DEVICE)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess
        mask = postprocess_mask(output)
        mask_img = Image.fromarray(mask, mode='L')
        
        # Convert to base64
        img_io = io.BytesIO()
        mask_img.save(img_io, 'PNG')
        img_io.seek(0)
        mask_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        return jsonify({
            "mask": mask_base64,
            "format": "png"
        })
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting server on port {port}...")
    serve(app, host='0.0.0.0', port=port)