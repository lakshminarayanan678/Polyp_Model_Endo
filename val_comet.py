from ultralytics import YOLO
from comet_ml import Experiment
import logging
import os
from glob import glob
import shutil

# Set your Comet credentials
COMET_API_KEY = "hE2ZXmFRrxO1jBzFIcUpDG0xU"
PROJECT_NAME = "benchmarking"
WORKSPACE = "lakshminarayanan-m"

# Initialize Comet experiment
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name=PROJECT_NAME,
    workspace=WORKSPACE,
    auto_output_logging="stdout",  # captures CLI prints
)

experiment.set_name("yolov12l-interdataset-test")
experiment.log_parameters({
    "model": "YOLOv12-L",
    "device": 0,
    "conf": 0.75,
    "iou": 0.5,
    "imgsz": 640,
    "batch": 1
})

# Run validation
model = YOLO("/home/endoai/Desktop/MODEL/YOLOv12l/runs/detect/yolov12-benchmark/weights/best.pt")
metrics = model.predict(
    "/home/endoai/Desktop/PAPER/TESTDATA/images/CVC-ClinicDB_6.png",
    conf=0.75,
    iou=0.5,
    save_json=True,
    device=0,
    plots=True,
    imgsz=640,
    batch=1,
    workers=8
)

# Compute F1
f1_class0 = 2 * metrics.box.p[0] * metrics.box.r[0] / (metrics.box.p[0] + metrics.box.r[0] + 1e-6)

# Print and log metrics
experiment.log_metrics({
    "Precision_class0": metrics.box.p[0],
    "Recall_class0": metrics.box.r[0],
    "F1_class0": metrics.box.f1[0],
    "F1_class0_manual": f1_class0,
    "AP50_class0": metrics.box.ap50[0],
    "AP75_class0": metrics.box.map75,
    "mAP_class0": metrics.box.map,
    "Model_fitness": metrics.box.fitness()
})

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

# 🔁 Automatically find latest val folder
val_dirs = sorted(glob("runs/detect/val*"), key=os.path.getmtime, reverse=True)
latest_val_dir = val_dirs[0] if val_dirs else None

# Log artifacts and plots (PR curve, confusion matrix, etc.)
if latest_val_dir:
    for img_file in glob(f"{latest_val_dir}/*.jpg"):
        experiment.log_image(img_file, step=0)
    for plot_file in glob(f"{latest_val_dir}/*.png"):
        experiment.log_image(plot_file, step=0)

    # Upload predictions.json if present
    pred_json = os.path.join(latest_val_dir, "predictions.json")
    if os.path.exists(pred_json):
        experiment.log_asset(pred_json)

experiment.end()
