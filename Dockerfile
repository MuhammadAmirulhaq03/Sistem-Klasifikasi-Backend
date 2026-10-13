FROM python:3.11-slim

# Install system dependencies untuk OpenCV & ONNX di Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install PyTorch CPU-only (~170MB vs 2.500MB CUDA)
# Menghemat 90% waktu download Docker!
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependensi Python lainnya
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy file aplikasi yang benar-benar dipakai
COPY . .

# Port Web Server & WebSocket Streaming
EXPOSE 8080

# Jalankan server utama
CMD ["python", "server.py"]
