import os
import json
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from tqdm import tqdm

TRAIN_DIR    = "train"                      
ANNOT_FILE   = "train/_annotations.coco.json"
OUTPUT_MODEL = "model_final.pth"

NUM_EPOCHS   = 10
BATCH_SIZE   = 2                            # Kurangi ke 2 jika GPU VRAM < 6GB
LEARNING_RATE = 0.005
NUM_WORKERS  = 4 if torch.cuda.is_available() else 0

class CocoDetectionDataset(Dataset):
    def __init__(self, img_dir, annot_path):
        with open(annot_path) as f:
            coco = json.load(f)

        # Buat mapping: image_id -> file_name
        self.img_dir   = img_dir
        self.id2file   = {img["id"]: img["file_name"] for img in coco["images"]}
        self.id2size   = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

        # Kumpulkan anotasi per gambar
        self.annots = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            if iid not in self.annots:
                self.annots[iid] = []
            self.annots[iid].append(ann)

        # Hanya proses gambar yang punya anotasi
        self.image_ids = [iid for iid in self.id2file if iid in self.annots]

        # Mapping category_id -> label (mulai dari 1, 0 = background)
        self.cat2label = {cat["id"]: idx + 1 for idx, cat in enumerate(coco["categories"])}
        self.label2name = {idx + 1: cat["name"] for idx, cat in enumerate(coco["categories"])}

        print(f"Dataset: {len(self.image_ids)} gambar")
        print(f"Kelas  : {self.label2name}")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        fname    = os.path.basename(self.id2file[image_id])
        img_path = os.path.join(self.img_dir, fname)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        boxes, labels = [], []
        for ann in self.annots[image_id]:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2label[ann["category_id"]])

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes):
    """Load Faster R-CNN pretrained, ganti head sesuai jumlah kelas."""
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # num_classes + 1 karena index 0 = background
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # Load dataset
    dataset = CocoDetectionDataset(TRAIN_DIR, ANNOT_FILE)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, collate_fn=collate_fn,
                         pin_memory=(device.type == "cuda"))

    num_classes = len(dataset.label2name)   # jumlah kelas tanpa background
    model = build_model(num_classes).to(device)

    # Optimizer hanya untuk parameter yang di-train
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print(f"\nMulai training {NUM_EPOCHS} epoch...\n")
    best_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for images, targets in pbar:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"  Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Simpan model terbaik
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  -> Model tersimpan: {OUTPUT_MODEL}  (loss terbaik: {best_loss:.4f})")

    print(f"\nTraining selesai! Model final: {OUTPUT_MODEL}")
    print(f"Kelas yang dilatih: {dataset.label2name}")
    # Simpan info kelas agar bisa dipakai di deteksi
    with open("label_map.json", "w") as f:
        json.dump(dataset.label2name, f, indent=2)
    print("Info kelas tersimpan: label_map.json")


if __name__ == "__main__":
    train()