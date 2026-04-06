# Lab 2: House Segmentation API with CI/CD Pipeline

## Overview

Dockerized house segmentation API using a custom-trained UNet model on aerial satellite imagery, served with Flask and Waitress. Built on top of Lab 1 with added secrets injection, CI/CD automation, and a fully trained segmentation model.

**Test Metrics:** IoU = 0.3284 | Dice = 0.4455

Docker Hub URL: https://hub.docker.com/r/mbelk181/house-segmentation-api

## Features

- UNet segmentation model trained on satellite building imagery
- Secrets management via `.env` + `python-dotenv`
- GitHub Actions CI/CD pipeline (test → build → push to Docker Hub)
- REST API with Flask and Waitress
- Binary pixel mask output (white = house, black = background)

## Quick Start

### Setup with Conda (Recommended)

```bash
conda create -n lab2 python=3.10 -y
conda activate lab2
conda install pytorch cpuonly -c pytorch -y
conda install flask numpy pillow waitress python-dotenv -c conda-forge -y
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### Run the API

```bash
python app.py
```

### Build and Run with Docker

```bash
docker-compose up --build
```

## API Endpoints

**GET /**

- Health check
- Returns model status and device info

**POST /predict**

- Input: multipart form with `image` field (PNG/JPG)
- Output: binary PNG mask image

**POST /predict_base64**

- Input: `{"image": "<base64_encoded_image>"}`
- Output: `{"mask": "<base64_encoded_mask>", "format": "png"}`

## Test the API

```bash
# Health check
curl http://localhost:5000/

# Predict mask from image
curl -X POST http://localhost:5000/predict \
  -F "image=@dataset/test/images/0000.png" \
  --output mask.png
```

## CI/CD Pipeline

GitHub Actions automatically:

1. Runs unit tests on every push
2. Builds the Docker image
3. Pushes to Docker Hub (on main branch)

Required GitHub secrets: `DOCKER_USERNAME`, `DOCKER_PASSWORD`

## Model

- **Architecture**: UNet (trained from scratch)
- **Dataset**: keremberke/satellite-building-segmentation (HuggingFace)
- **Training**: 30 epochs, Adam optimizer, BCE loss, T4 GPU
- **Input**: 256×256 RGB aerial image
- **Output**: 256×256 binary segmentation mask
- **Test IoU**: 0.3284 | **Test Dice**: 0.4455

## Project Structure

```
├── app.py                  # Flask API server
├── train.py                # Model training script
├── evaluate.py             # Evaluation and metrics
├── dataset.py              # Dataset loading
├── prepare_masks.py        # Pixel mask generation (Week 7)
├── model/
│   └── unet.py             # UNet architecture
├── dataset/
│   ├── train/              # Training images and masks
│   ├── val/                # Validation images and masks
│   └── test/               # Test images and masks
├── tests/
│   └── test_api.py         # API unit tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
