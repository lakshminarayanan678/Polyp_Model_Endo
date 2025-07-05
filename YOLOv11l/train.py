from ultralytics import YOLO

# Load a model from YAML
model = YOLO("yolo11l.yaml")  # or "yolo11l.pt" if pretrained

# Train the model
results = model.train(
    data="/home/endoai/Desktop/MODEL/YOLOv11l/data.yaml",
    batch=16,
    epochs=30,
    workers=8,
    imgsz=640,
    val=True,          # evaluates on val set after each epoch
    plots=True,        # saves training/val loss & metric curves
    save=True,
    name="yolov11l-benchmark"
)