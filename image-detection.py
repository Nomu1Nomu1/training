import cv2
import json
import time
import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH       = "model_final.pth"
LABEL_MAP_PATH   = "label_map.json"
CONFIDENCE_THRESH = 0.5       # Minimum confidence untuk tampilkan kotak
CAMERA_WIDTH     = 640        # Resolusi kamera (lebih kecil = lebih cepat)
CAMERA_HEIGHT    = 480

# Warna per kelas (BGR format OpenCV)
BOX_COLORS = [
    (0, 255, 80),    # Hijau
    (0, 100, 255),   # Oranye
    (255, 50, 50),   # Biru
    (255, 0, 200),   # Magenta
    (0, 255, 255),   # Kuning
]


def load_model(model_path, label_map_path):
    with open(label_map_path) as f:
        raw = json.load(f)
    # label_map key bisa string (dari JSON), konversi ke int
    label_map = {int(k): v for k, v in raw.items()}
    num_classes = len(label_map)

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"Model dimuat: {model_path}")
    print(f"Kelas: {label_map}")
    return model, label_map


def open_camera():
    """
    Coba buka Arducam/PiCamera lewat picamera2 dulu,
    fallback ke OpenCV (webcam USB / device index 0).
    """
    try:
        from picamera2 import Picamera2
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        print("Kamera: Arducam/PiCamera2 aktif")
        return picam, "picamera2"
    except Exception as e:
        print(f"PiCamera2 gagal ({e}), coba OpenCV...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError("Tidak bisa membuka kamera!")
        print("Kamera: OpenCV (device 0) aktif")
        return cap, "opencv"


def read_frame(cam, cam_type):
    if cam_type == "picamera2":
        frame_rgb = cam.capture_array()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None
        return frame


to_tensor = transforms.ToTensor()

def detect(frame_bgr, model):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(Image.fromarray(rgb)).unsqueeze(0)

    with torch.no_grad():
        preds = model(tensor)[0]

    results = []
    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score.item() < CONFIDENCE_THRESH:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        results.append({
            "box":   (x1, y1, x2, y2),
            "label": label.item(),
            "score": score.item(),
        })
    return results


def draw_detections(frame, detections, label_map):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label_id = det["label"]
        score    = det["score"]
        name     = label_map.get(label_id, f"kelas-{label_id}")

        color_idx = (label_id - 1) % len(BOX_COLORS)
        color     = BOX_COLORS[color_idx]

        # Kotak deteksi
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Background label
        text  = f"{name}: {score*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

        # Teks label
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame


def run():
    model, label_map = load_model(MODEL_PATH, LABEL_MAP_PATH)

    cam, cam_type = open_camera()
    print("\nTekan 'q' untuk keluar | 's' untuk screenshot")

    fps_counter, fps_display = 0, 0.0
    t_start = time.time()

    try:
        while True:
            frame = read_frame(cam, cam_type)
            if frame is None:
                print("[ERROR] Gagal membaca frame")
                break

            detections = detect(frame, model)
            frame = draw_detections(frame, detections, label_map)
            frame = draw_fps(frame, fps_display)

            cv2.imshow("AI Detection - Tekan q untuk keluar", frame)

            # Hitung FPS
            fps_counter += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_start = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Screenshot: {fname}")

    finally:
        if cam_type == "picamera2":
            cam.stop()
        else:
            cam.release()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    run()