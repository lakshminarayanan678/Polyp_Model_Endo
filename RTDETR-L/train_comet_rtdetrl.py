from comet_ml import Experiment
from ultralytics import RTDETR
import os
import numpy as np

# ==== COMET CONFIGURATION ====
experiment = Experiment(
    api_key="hE2ZXmFRrxO1jBzFIcUpDG0xU",
    project_name="benchmarking",
    workspace="lakshminarayanan-m"
)

# ==== EXPERIMENT INFO ====
experiment.set_name("rtdetr-l-benchmark")

# ==== TRAINING CONFIG ====
model_config = "rtdetr-l.yaml"
data_config = "/home/endoai/Desktop/MODEL/RTDETR-L/data.yaml"
batch_size = 16
epochs = 30
workers = 8
image_size = 640
run_name = "rtdetr-l-benchmark"
save_results = True
plot_results = True

# ==== LOG HYPERPARAMETERS ====
experiment.log_parameters({
    "model": model_config,
    "data": data_config,
    "batch": batch_size,
    "epochs": epochs,
    "workers": workers,
    "imgsz": image_size,
    "run_name": run_name
})

# ==== LOAD MODEL ====
model = RTDETR(model_config)

# ==== TRAIN ====
results = model.train(
    data=data_config,
    batch=batch_size,
    epochs=epochs,
    workers=workers,
    imgsz=image_size,
    val=True,
    plots=plot_results,
    save=save_results,
    name=run_name
)

# ==== LOG TRAINING CURVES ====
plot_dir = f"runs/detect/{run_name}"
for plot_file in ["results.png", "F1_curve.png", "confusion_matrix.png"]:
    plot_path = os.path.join(plot_dir, plot_file)
    if os.path.exists(plot_path):
        experiment.log_image(plot_path, step=epochs)

# ==== FINAL VALIDATION METRICS ====
try:
    val_metrics = results.metrics.box.mean_results().cpu().numpy()
    precision, recall, mAP50, mAP75 = val_metrics[:4]
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    experiment.log_metrics({
        "val/precision": precision,
        "val/recall": recall,
        "val/f1": f1,
        "val/mAP50": mAP50,
        "val/mAP75": mAP75,
        "val/IoU@0.5": mAP50,
        "val/IoU@0.75": mAP75
    })
except Exception as e:
    print(f"[WARNING] Validation metrics logging failed: {e}")

# ==== TEST METRICS ====
try:
    test_results = model.val(data=data_config, split="test")
    test_metrics = test_results.metrics.box.mean_results().cpu().numpy()
    p, r, m50, m75 = test_metrics[:4]
    f1 = 2 * p * r / (p + r + 1e-6)

    experiment.log_metrics({
        "test/precision": p,
        "test/recall": r,
        "test/f1": f1,
        "test/mAP50": m50,
        "test/mAP75": m75,
        "test/IoU@0.5": m50,
        "test/IoU@0.75": m75
    })
except Exception as e:
    print(f"[WARNING] Test metric logging failed: {e}")

# ==== LOG MODEL ARTIFACT ====
best_model_path = os.path.join(plot_dir, "weights", "best.pt")
if os.path.exists(best_model_path):
    experiment.log_model("best_model", best_model_path)

# ==== END COMET EXPERIMENT ====
experiment.end()
