from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="data/india_roads_final/data.yaml",
    epochs=15,       # reduced from 50 — CPU is slow, and fine-tuning needs fewer epochs anyway
    imgsz=416,        # smaller than 640 — meaningfully faster on CPU, still fine for accuracy
    batch=8,          # smaller batch — CPU can't handle 16 efficiently
    device="cpu",
    patience=5,
    project="data/weights",
    name="india_finetune",
)