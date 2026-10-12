# Sistem Klasifikasi & Deteksi Brondolan Sawit (Backend Engine)

Backend API dan Real-time Streaming Server untuk sistem klasifikasi kematangan brondolan kelapa sawit menggunakan YOLOv8, ONNX Runtime, Flask, dan WebSocket.

---

## 🚀 Cara Menjalankan dengan Docker (Rekomendasi)

Siapa pun yang men-clone repositori ini **hanya membutuhkan Docker**. Tidak perlu menginstall Python, CUDA, ataupun XAMPP/MySQL di laptop/komputer.

### 1. Clone Repositori
```bash
git clone https://github.com/MuhammadAmirulhaq03/Sistem-Klasifikasi-Backend.git
cd Sistem-Klasifikasi-Backend
```

### 2. Jalankan dengan Docker Compose
```bash
docker compose up -d --build
```

Setelah perintah di atas selesai:
* **Backend API & WebSocket:** Berjalan di `http://localhost:8080`
* **Database MySQL:** Otomatis aktif di port `3306` (Database: `db_brondolan`)
* **Live Video Streaming:** Langsung memancarkan hasil deteksi secara mulus ke frontend dashboard.

Untuk mematikan container:
```bash
docker compose down
```

---

## 💻 Cara Menjalankan Manual (Tanpa Docker)

Jika ingin menjalankan langsung di Python lokal:

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Siapkan MySQL
Pastikan MySQL (misal XAMPP) sudah aktif di `localhost:3306` dengan database `db_brondolan`.

### 3. Jalankan Server
```bash
python server.py
```

---

## 📡 Daftar Endpoint API

| Metode | Endpoint | Keterangan |
| :---: | :--- | :--- |
| **GET** | `/api/detections` | Mengambil seluruh riwayat hasil deteksi dari database |
| **GET** | `/api/detections/last` | Mengambil 1 deteksi terakhir |
| **GET** | `/api/frame-info` | Informasi FPS, latensi inferensi, dan timestamp |
| **GET** | `/video_feed` | Live MJPEG Video Stream (bisa langsung dibuka di browser atau tag `<img>`) |
| **WS** | `socket.io` (Event: `stream`) | Real-time WebSocket stream untuk dashboard web (base64 frame + boxes) |

---

## 🧠 Model Deteksi
* **`best_int8.onnx`**: Model teroptimasi INT8 (41 MB) yang berjalan sangat cepat dan efisien pada prosesor (CPU).
* **`best.pt`**: Bobot asli model master YOLOv8 PyTorch.
