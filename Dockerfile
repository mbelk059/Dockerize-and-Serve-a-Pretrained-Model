FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .
COPY model/ ./model/
COPY saved_model.pth .

EXPOSE 5000

CMD ["python", "app.py"]