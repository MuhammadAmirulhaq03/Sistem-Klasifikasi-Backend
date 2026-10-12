import cv2
import time
import threading
import deteksi

VIDEO_PATH = "brondolan.mp4"
FPS_LIMIT = 15

running = True


def video_stream_loop():
    global running

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Video tidak bisa dibuka")
        return

    print("[INFO] Video stream started")

    frame_delay = 1.0 / FPS_LIMIT

    while running:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        deteksi.latest_frame = frame

        time.sleep(frame_delay)

    cap.release()
    print("[INFO] Video stream stopped")


def start_stream():
    thread = threading.Thread(target=video_stream_loop, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    start_stream()
    deteksi.main()