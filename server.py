from flask import Flask, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO
from build_db import BrondolanDB
import shared_state
import deteksi
import threading
import cv2
import time

app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

db = BrondolanDB()

connected_clients = 0
clients_lock = threading.Lock()


# ─────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────

@app.route("/api/detections", methods=["GET"])
def get_detections():
    rows = db.fetch_all()
    return jsonify([
        {
            "id": r[0],
            "id_deteksi": r[1],
            "class": r[2],
            "boundingbox": r[3],
            "timestamp": str(r[4])
        }
        for r in rows
    ])


@app.route("/api/detections/last", methods=["GET"])
def get_last_detection():
    rows = db.fetch_last_n(1)
    if not rows:
        return jsonify({})
    r = rows[0]
    return jsonify({
        "id": r[0],
        "id_deteksi": r[1],
        "class": r[2],
        "boundingbox": r[3],
        "timestamp": str(r[4])
    })


@app.route("/api/frame-info", methods=["GET"])
def get_frame_info():
    with shared_state.lock:
        fps = shared_state.current_fps
        latency = shared_state.current_latency
        timestamp = shared_state.latest_frame_timestamp
        yolo_fps = getattr(shared_state, "yolo_fps", 0)

    return jsonify({
        "fps": fps,
        "yolo_fps": yolo_fps,
        "latency_ms": latency,
        "timestamp": timestamp
    })


# ─────────────────────────────────────────
# MJPEG STREAM (untuk preview langsung di browser via tag <img>)
# ─────────────────────────────────────────

def generate_mjpeg():
    """
    Streaming video MJPEG murni (HTTP Multipart) langsung dari shared_state.latest_frame.
    """
    while True:
        with shared_state.lock:
            frame = shared_state.latest_frame

        if frame is None:
            time.sleep(0.04)
            continue

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.04)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────
# WEBSOCKET STREAM
# ─────────────────────────────────────────

def websocket_stream():
    """
    Emit ke client saat frame stream baru siap.
    Hanya emit jika ada client aktif yang terhubung agar hemat CPU.
    """
    global connected_clients

    while True:
        triggered = deteksi.frame_ready_event.wait(timeout=2.0)
        if not triggered:
            continue

        deteksi.frame_ready_event.clear()

        # Cek apakah ada client terhubung
        with clients_lock:
            has_client = (connected_clients > 0)

        if not has_client:
            continue

        with shared_state.lock:
            encoded = shared_state.latest_encoded_frame
            boxes = shared_state.latest_boxes
            timestamp = shared_state.latest_frame_timestamp
            fps = shared_state.current_fps
            latency = shared_state.current_latency

        if encoded is None:
            continue

        socketio.emit("stream", {
            "image": encoded,
            "boxes": boxes,
            "timestamp": timestamp,
            "fps": fps,
            "latency": latency
        })


# ─────────────────────────────────────────
# SOCKET EVENTS
# ─────────────────────────────────────────

@socketio.on("connect")
def handle_connect():
    global connected_clients
    with clients_lock:
        connected_clients += 1
    print(f"[WS] Client connected (Total: {connected_clients})")

    with shared_state.lock:
        encoded = shared_state.latest_encoded_frame
        boxes = shared_state.latest_boxes
        timestamp = shared_state.latest_frame_timestamp
        fps = shared_state.current_fps
        latency = shared_state.current_latency

    if encoded is not None:
        socketio.emit("stream", {
            "image": encoded,
            "boxes": boxes,
            "timestamp": timestamp,
            "fps": fps,
            "latency": latency
        })


@socketio.on("disconnect")
def handle_disconnect():
    global connected_clients
    with clients_lock:
        connected_clients = max(0, connected_clients - 1)
    print(f"[WS] Client disconnected (Total: {connected_clients})")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Server running on port 8080")

    deteksi.main()

    stream_thread = threading.Thread(
        target=websocket_stream,
        daemon=True,
        name="WebSocketStream"
    )
    stream_thread.start()

    socketio.run(
        app,
        host="0.0.0.0",
        port=8080,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )