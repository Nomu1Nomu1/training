import cv2
import json
import time
import threading
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

MODEL_PATH        = "model_final.pth"
LABEL_MAP_PATH    = "label_map.json"
CONFIDENCE_THRESH = 0.5

CAMERA_WIDTH      = 640
CAMERA_HEIGHT     = 480

# Resolusi input ke model
INFER_WIDTH       = 320
INFER_HEIGHT      = 320

DETECT_EVERY_N    = 2   # Naik ke 3++ jika CPU masih berat

BOX_COLORS = [
    (0, 255, 80),
    (0, 100, 255),
    (255, 50, 50),
    (255, 0, 200),
    (0, 255, 255),
]

torch.set_num_threads(4)                     
torch.backends.cudnn.benchmark = True

def load_model(model_path, label_map_path):
    with open(label_map_path) as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    num_classes = len(label_map)

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_half = device.type == "cuda"
    if use_half:
        model.half()
        print("Mode: GPU + FP16")
    else:
        print("Mode: CPU FP32")

    print(f"Model dimuat: {model_path}")
    print(f"Kelas: {label_map}")
    return model, label_map, device, use_half

class FrameReader:
    """
    Membaca frame di thread terpisah agar loop utama tidak menunggu kamera.
    """
    def __init__(self):
        self.cam, self.cam_type = self._open_camera()
        self.frame = None
        self.lock  = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open_camera(self):
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
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Minimalkan buffer lag
            if not cap.isOpened():
                raise RuntimeError("Tidak bisa membuka kamera!")
            print("Kamera: OpenCV (device 0) aktif")
            return cap, "opencv"

    def _reader(self):
        while not self._stop:
            if self.cam_type == "picamera2":
                frame_rgb = self.cam.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.cam.read()
                if not ret:
                    continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)
        if self.cam_type == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()

_to_tensor = transforms.ToTensor()

def detect(frame_bgr, model, device, use_half):
    """
    Resize frame ke INFER_WIDTH x INFER_HEIGHT sebelum inferensi,
    lalu scale bounding box kembali ke ukuran frame asli.
    """
    h_orig, w_orig = frame_bgr.shape[:2]

    # Resize untuk model
    small = cv2.resize(frame_bgr, (INFER_WIDTH, INFER_HEIGHT))
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = _to_tensor(Image.fromarray(rgb)).unsqueeze(0).to(device)
    if use_half:
        tensor = tensor.half()

    with torch.no_grad():
        preds = model(tensor)[0]

    # Faktor scale balik ke resolusi asli
    sx = w_orig / INFER_WIDTH
    sy = h_orig / INFER_HEIGHT

    results = []
    for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
        if score.item() < CONFIDENCE_THRESH:
            continue
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "box":   (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
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

        color = BOX_COLORS[(label_id - 1) % len(BOX_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text  = f"{name}: {score*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return frame

def run():
    model, label_map, device, use_half = load_model(MODEL_PATH, LABEL_MAP_PATH)
    reader = FrameReader()

    print("\nTekan 'q' untuk keluar | 's' untuk screenshot")

    fps_counter  = 0
    fps_display  = 0.0
    t_start      = time.time()
    frame_idx    = 0
    last_dets    = []   # Simpan hasil deteksi terakhir

    try:
        while True:
            frame = reader.read()
            if frame is None:
                time.sleep(0.005)
                continue

            if frame_idx % DETECT_EVERY_N == 0:
                last_dets = detect(frame, model, device, use_half)
            frame_idx += 1

            frame = draw_detections(frame, last_dets, label_map)
            frame = draw_fps(frame, fps_display)

            cv2.imshow("AI Detection - Tekan q untuk keluar", frame)

            # Hitung FPS berdasarkan frame yang di-render
            fps_counter += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_start     = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Screenshot: {fname}")

    finally:
        reader.stop()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    run()