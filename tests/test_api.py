import pytest
import io
from PIL import Image
import numpy as np

def get_app():
    import app
    app.app.config["TESTING"] = True
    return app.app.test_client()

def test_health():
    """Test health check endpoint"""
    client = get_app()
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "House Segmentation" in data["message"]

def test_predict_no_image():
    """Test predict endpoint without image"""
    client = get_app()
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_predict_with_image():
    """Test predict endpoint with valid image"""
    client = get_app()
    # Create a test image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"image": (buf, "test.png")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.mimetype == "image/png"

def test_predict_base64():
    """Test base64 prediction endpoint"""
    import base64
    client = get_app()
    
    # Create test image and encode to base64
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    response = client.post(
        "/predict_base64",
        json={"image": img_base64}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "mask" in data
    assert data["format"] == "png"