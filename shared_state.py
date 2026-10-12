import threading

lock = threading.Lock()

latest_frame = None
latest_encoded_frame = None
latest_boxes = []
current_fps = 0
current_latency = 0.0
latest_frame_timestamp = None
yolo_fps = 0
raw_frame = None