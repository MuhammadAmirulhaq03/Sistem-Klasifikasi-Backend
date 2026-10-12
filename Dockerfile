FROM python:3.11-slim

# Install system dependencies untuk OpenCV & ONNX di Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file aplikasi
COPY . .

# Port Web Server & WebSocket Streaming
EXPOSE 8080

# Jalankan server utama
CMD ["python", "server.py"]
