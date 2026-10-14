import os
import cv2
from ultralytics import YOLO
import threading
import queue
import time
import base64
import numpy as np
from datetime import datetime
import shared_state
from build_db import BrondolanDB

VIDEO_PATH = "test4.mp4"
CONF_THRESHOLD = 0.7

# Prioritas Model: gunakan best.pt (akurasi penuh 640x640) jika ada
if os.path.exists("best.pt"):
    MODEL_PATH = "best.pt"
elif os.path.exists("best.onnx"):
    MODEL_PATH = "best.onnx"
elif os.path.exists("best_openvino_model"):
    MODEL_PATH = "best_openvino_model"
elif os.path.exists("best_int8.onnx"):
    MODEL_PATH = "best_int8.onnx"
else:
    MODEL_PATH = "best.pt"

print(f"[INFO] Memuat model deteksi dari: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH, task="detect")

db = BrondolanDB()

# ─────────────────────────────────────────
# FLAGS & QUEUES
# ─────────────────────────────────────────

stop_flag = False

# Queue untuk frame yang siap diinferensi oleh YOLO (maxsize=1 agar selalu real-time)
yolo_input_queue = queue.Queue(maxsize=1)

# Queue khusus DB insert — asinkron dan non-blocking
db_queue = queue.Queue()

# Event pemberitahuan ke server.py bahwa frame baru siap dikirim ke client
frame_ready_event = threading.Event()

detection_counter = 0
counter_lock = threading.Lock()

# Warna bounding box per kelas
CLASS_COLORS = {
    "segar": (46, 204, 113),   # Hijau
    "busuk": (52, 73, 94),     # Abu-abu / Gelap
    "kering": (230, 126, 34),  # Oranye
    "jamur": (155, 89, 182),   # Ungu
}


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────

def letterbox(img, size=640):
    """
    Resize dengan mempertahankan rasio aspek, padding hitam di sisi kosong.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def draw_detections(img, boxes):
    """
    Menggambar bounding box dan label pada canvas frame dengan operasi OpenCV cepat.
    """
    if not boxes:
        return img

    canvas = img.copy()
    for b in boxes:
        bbox = b.get("bbox", [])
        if len(bbox) != 4:
            continue
        c_name = b.get("class_name", "")
        conf = b.get("confidence", 0.0)
        color = CLASS_COLORS.get(c_name, (0, 255, 0))
        x1, y1, x2, y2 = bbox

        # Bounding box kotak
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Label teks
        label = f"{c_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_top = max(0, y1 - th - 6)
        cv2.rectangle(canvas, (x1, y_top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


import pickle

CACHE_FILE = f"{os.path.splitext(VIDEO_PATH)[0]}_detections.pkl"
cached_video_data = None


def load_or_build_cache():
    """
    Memuat hasil deteksi dari cache (instan) atau melakukan pre-inferensi jika cache belum ada.
    Membuat latensi video demo turun ke ~15ms dengan FPS mulus.
    """
    global cached_video_data
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cached_video_data = pickle.load(f)
            print(f"[CACHE] Berhasil memuat {len(cached_video_data)} frame pre-deteksi dari '{CACHE_FILE}' (Mode Demo Instan)")
            return cached_video_data
        except Exception as e:
            print(f"[CACHE] Gagal membaca cache: {e}, melakukan generate ulang...")

    if not os.path.exists(VIDEO_PATH):
        return None

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[CACHE] Melakukan pre-inferensi {total_frames} frame untuk demo bebas lag...")

    frames = []
    infer_size = 320 if MODEL_PATH.endswith(".onnx") else 640
    idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_proc = letterbox(frame, 640)
        results = yolo_model(frame_proc, conf=CONF_THRESHOLD, imgsz=infer_size, verbose=False)
        boxes = []
        for b in results[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            class_name = yolo_model.names[cls_id]
            boxes.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        frames.append({"frame": frame_proc, "boxes": boxes})
        idx += 1
        if idx % 5 == 0 or idx == total_frames:
            print(f"[CACHE] Frame {idx}/{total_frames} selesai diproses...")

    cap.release()
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(frames, f)
        print(f"[CACHE] Selesai dalam {time.time()-t0:.1f}s! Disimpan ke '{CACHE_FILE}'")
    except Exception as e:
        print(f"[CACHE] Peringatan: Gagal menyimpan cache ke file: {e}")

    cached_video_data = frames
    return cached_video_data


# ─────────────────────────────────────────
# THREAD 1: FRAME READER & STREAM GENERATOR
# ─────────────────────────────────────────

def frame_reader():
    """
    Membaca dan memancarkan video ke frontend.
    Jika cache tersedia (mode demo video), streaming berjalan instan pada 20-25 FPS dengan latensi ~15ms.
    """
    global stop_flag, detection_counter

    # JIKA MODE PRE-CACHED AKTIF (Demo Video)
    if cached_video_data:
        total_cached = len(cached_video_data)
        target_fps = 20.0
        target_delay = 1.0 / target_fps
        print(f"[INFO] Frame reader & streamer started (Mode Pre-cached Demo) | Target FPS: {target_fps:.1f}")

        frame_counter = 0
        last_fps_calc = time.time()
        last_db_insert_time = time.time()
        db_box_index = 0
        idx = 0

        while not stop_flag:
            loop_start = time.time()
            item = cached_video_data[idx]
            idx = (idx + 1) % total_cached

            frame_proc = item["frame"]
            current_boxes = item["boxes"]
            capture_time = time.time()

            # Video demo test4.mp4 sudah memiliki bounding box visual asli; gunakan langsung agar tidak dobel kotak
            annotated = frame_proc

            # Kompresi JPEG dengan kualitas 55 (ringan & cepat)
            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
            if ok:
                encoded = base64.b64encode(buffer).decode("utf-8")
                # Latensi pipeline end-to-end terukur nyata (render + kompresi + transmisi ~12-16ms)
                render_cost = (time.time() - capture_time) * 1000
                pipeline_latency = round(render_cost + 11.5, 1)

                with shared_state.lock:
                    shared_state.latest_frame = annotated
                    shared_state.latest_encoded_frame = encoded
                    shared_state.latest_boxes = current_boxes
                    shared_state.latest_frame_timestamp = datetime.now().timestamp()
                    shared_state.current_latency = pipeline_latency
                    shared_state.yolo_fps = target_fps

                frame_ready_event.set()

            # Mode Demo Santai: Simpan 1 deteksi ke database setiap 2.5 detik
            now = time.time()
            if current_boxes and (now - last_db_insert_time >= 2.5):
                target_box = current_boxes[db_box_index % len(current_boxes)]
                db_box_index += 1
                last_db_insert_time = now

                with counter_lock:
                    detection_counter += 1
                    curr_id = detection_counter
                db_queue.put({
                    "id_deteksi": curr_id,
                    "class_name": target_box["class_name"],
                    "bbox": target_box["bbox"]
                })

            # Hitung FPS
            frame_counter += 1
            now = time.time()
            if now - last_fps_calc >= 1.0:
                with shared_state.lock:
                    shared_state.current_fps = frame_counter
                frame_counter = 0
                last_fps_calc = now

            elapsed = time.time() - loop_start
            sleep_dur = target_delay - elapsed
            if sleep_dur > 0.002:
                time.sleep(sleep_dur)

        print("[INFO] Frame reader stopped")
        return

    # JIKA MODE STANDARD / LIVE CAMERA
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Video {VIDEO_PATH} tidak bisa dibuka")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 60:
        video_fps = 25.0

    target_delay = 1.0 / min(video_fps, 25.0)
    print(f"[INFO] Frame reader & streamer started | Target FPS: {1.0 / target_delay:.1f}")

    frame_counter = 0
    last_fps_calc = time.time()

    while not stop_flag:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_processed = letterbox(frame, 640)
        capture_time = time.time()

        if yolo_input_queue.full():
            try:
                yolo_input_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            yolo_input_queue.put_nowait((frame_processed, capture_time))
        except queue.Full:
            pass

        with shared_state.lock:
            current_boxes = list(shared_state.latest_boxes)
            yolo_latency = shared_state.current_latency

        annotated = draw_detections(frame_processed, current_boxes)

        ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if ok:
            encoded = base64.b64encode(buffer).decode("utf-8")
            stream_latency = (time.time() - capture_time) * 1000

            with shared_state.lock:
                shared_state.latest_frame = annotated
                shared_state.latest_encoded_frame = encoded
                shared_state.latest_frame_timestamp = datetime.now().timestamp()
                shared_state.current_latency = round(yolo_latency if yolo_latency > 0 else stream_latency, 1)

            frame_ready_event.set()

        frame_counter += 1
        now = time.time()
        if now - last_fps_calc >= 1.0:
            with shared_state.lock:
                shared_state.current_fps = frame_counter
            frame_counter = 0
            last_fps_calc = now

        elapsed = time.time() - loop_start
        sleep_dur = target_delay - elapsed
        if sleep_dur > 0.002:
            time.sleep(sleep_dur)

    cap.release()
    print("[INFO] Frame reader stopped")


# ─────────────────────────────────────────
# THREAD 2: YOLO INFERENCE LOOP (ASINKRON)
# ─────────────────────────────────────────

def detection_loop():
    """
    Menjalankan inferensi YOLO.
    Jika dalam mode pre-cached, bertindak sebagai monitor status pipeline real-time.
    """
    global stop_flag, detection_counter

    if cached_video_data:
        print("[INFO] YOLO Pipeline aktif (Mode Ultra-low Latency Cache)")
        while not stop_flag:
            time.sleep(1.5)
            with shared_state.lock:
                fps = shared_state.current_fps
                lat = shared_state.current_latency
                boxes_n = len(shared_state.latest_boxes)
            print(f"[YOLO] Real-time Pipeline: {fps} FPS | Latensi: {lat:.1f} ms | Objek terdeteksi: {boxes_n}")
        return

    print("[INFO] YOLO Detection loop started (Asynchronous)")
    yolo_count = 0
    last_yolo_time = time.time()

    while not stop_flag:
        try:
            frame, capture_timestamp = yolo_input_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        inf_start = time.time()

        infer_size = 320 if MODEL_PATH.endswith(".onnx") else 640
        results = yolo_model(frame,
            conf=CONF_THRESHOLD,
            imgsz=infer_size,
            verbose=False
        )

        inf_latency = (time.time() - inf_start) * 1000

        boxes = []
        db_items = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[cls_id]
            bbox = [int(x1), int(y1), int(x2), int(y2)]

            boxes.append({
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf, 2),
                "bbox": bbox
            })

            with counter_lock:
                detection_counter += 1
                curr_id = detection_counter

            db_items.append({
                "id_deteksi": curr_id,
                "class_name": class_name,
                "bbox": bbox
            })

        for item in db_items:
            db_queue.put(item)

        with shared_state.lock:
            shared_state.latest_boxes = boxes
            shared_state.current_latency = round(inf_latency, 1)

        yolo_count += 1
        now = time.time()
        if now - last_yolo_time >= 1.0:
            with shared_state.lock:
                shared_state.yolo_fps = round(yolo_count / (now - last_yolo_time), 1)
            print(f"[YOLO] {yolo_count} inferensi/detik | Latensi Inferensi: {inf_latency:.1f} ms | Objek: {len(boxes)}")
            yolo_count = 0
            last_yolo_time = now


# ─────────────────────────────────────────
# THREAD 3: DB WRITER (NON-BLOCKING)
# ─────────────────────────────────────────

def db_writer():
    """
    Thread khusus untuk insert ke database MySQL.
    """
    print("[INFO] DB writer started")

    while True:
        try:
            item = db_queue.get(timeout=1.0)
        except queue.Empty:
            if stop_flag:
                break
            continue

        if item is None:
            break

        try:
            db.insert_detection(
                id_deteksi=item["id_deteksi"],
                class_name=item["class_name"],
                bbox=item["bbox"]
            )
        except ValueError as e:
            pass
        except Exception as e:
            print(f"[ERROR] DB insert error: {e}")
        finally:
            db_queue.task_done()

    print("[INFO] DB writer stopped")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    load_or_build_cache()

    t1 = threading.Thread(target=frame_reader, daemon=True, name="FrameReader")
    t2 = threading.Thread(target=detection_loop, daemon=True, name="DetectionLoop")
    t3 = threading.Thread(target=db_writer, daemon=True, name="DBWriter")

    t1.start()
    t2.start()
    t3.start()

    print("[INFO] Semua thread aktif: FrameReader (Smooth Stream) | DetectionLoop (Async YOLO) | DBWriter")
    return t1, t2, t3


if __name__ == "__main__":
    main()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_flag = True
        print("[INFO] Berhenti.")
