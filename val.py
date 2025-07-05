from ultralytics import YOLO
from ultralytics import RTDETR
import logging

# Load a model
model = YOLO("/home/endoai/Desktop/MODEL/YOLOv11l/runs/detect/yolov11l-benchmark/weights/best.pt")
metrics = model.val(data="/home/endoai/Desktop/MODEL/YOLOv11l/data.yaml", split= 'test', conf=0.75, iou=0.5, save_json=True, device=0, plots=True, imgsz=640, batch=1, workers=8)

# from ultralytics.utils.benchmarks import benchmark

# # Benchmark on GPU
# benchmark(model="/home/endoai/Desktop/MODEL/RTDETR-L/runs/detect/rtdetr-l-benchmark/weights/best.pt", data="/home/endoai/Desktop/MODEL/test_etis.yaml", imgsz=640, half=False)

f1_class0 = 2 * metrics.box.p[0] * metrics.box.r[0] / (metrics.box.p[0] + metrics.box.r[0] + 1e-6)
print("\n===== DETECTION METRICS =====")
print(f"Precision     : {metrics.box.p[0]:.4f}")
print(f"Recall        : {metrics.box.r[0]:.4f}")
print(f"F1 Score      : {metrics.box.f1[0]:.4f}")
print(f"F1CLASS_form       : {f1_class0:.4f}")
print(f"AP@0.5       : {metrics.box.ap50[0]:.4f}")
print(f"AP@0.75      : {metrics.box.map75:.4f}")
print(f"AP@0.5:0.95  : {metrics.box.map:.4f}")

print("\n===== DETECTION METRICS =====")
print(f"Precision (mean)   : {metrics.box.mp:.4f}")       # mean precision
print(f"Recall (mean)      : {metrics.box.mr:.4f}")       # mean recall
print(f"F1 Score (polyp)   : {metrics.box.f1.mean():.4f}")
print(f"AP@0.5 (mean)      : {metrics.box.ap50.mean():.4f}")
print(f"mAP@0.5 (mean)      : {metrics.box.map50:.4f}")
try:
    print(f"Model_fitness      : {metrics.box.fitness():.4f}")
except Exception as e:
    logging.error(f"Fitness error: {e}")