from flask import Flask, request, jsonify
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load pretrained sentiment analysis model from HuggingFace
logger.info("Loading pretrained model...")
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
logger.info("Model loaded successfully!")

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Sentiment Analysis API",
        "model": "distilbert-base-uncased-finetuned-sst-2-english"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        result = model(text)
        
        return jsonify({
            "input_text": text,
            "prediction": result[0]
        })
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    logger.info("Starting server on port 5000...")
    serve(app, host='0.0.0.0', port=5000)