from comet_ml import Experiment
from ultralytics import YOLO
import os
import numpy as np

# ==== COMET CONFIGURATION ====
experiment = Experiment(
    api_key="hE2ZXmFRrxO1jBzFIcUpDG0xU",
    project_name="benchmarking",
    workspace="lakshminarayanan-m"
)

# ==== EXPERIMENT INFO ====
experiment.set_name("yolov11-benchmark")

# ==== TRAINING CONFIG ====
model_config = "yolo11l.yaml"
data_config = "/home/endoai/Desktop/MODEL/YOLOv11l/data.yaml"
batch_size = 16
epochs = 30
workers = 8
image_size = 640
run_name = "yolov11l-benchmark"
save_results = True
plot_results = True

# ==== LOG HYPERPARAMETERS TO COMET ====
experiment.log_parameters({
    "model": model_config,
    "data": data_config,
    "batch": batch_size,
    "epochs": epochs,
    "workers": workers,
    "imgsz": image_size,
    "run_name": run_name
})

# ==== LOAD AND TRAIN MODEL ====
model = YOLO(model_config)  # Build new model from YAML

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
# If available: results.history is [dict, dict, ...] over epochs
if hasattr(results, "metrics") and hasattr(results, "training_results"):
    for epoch, metrics in enumerate(results.training_results):
        experiment.log_metrics({
            "train/box_loss": metrics.get("box_loss", np.nan),
            "train/cls_loss": metrics.get("cls_loss", np.nan),
            "train/obj_loss": metrics.get("obj_loss", np.nan),
        }, step=epoch)

        if "val/precision" in metrics:
            experiment.log_metrics({
                "val/precision": metrics["val/precision"],
                "val/recall": metrics["val/recall"],
                "val/mAP50": metrics["val/mAP50"],
                "val/mAP75": metrics.get("val/mAP75", np.nan),
                "val/f1": 2 * metrics["val/precision"] * metrics["val/recall"] / (metrics["val/precision"] + metrics["val/recall"] + 1e-6),
            }, step=epoch)

# ==== LOG FINAL VALIDATION METRICS ====
try:
    final_metrics = results.metrics  # dict or object
    experiment.log_metrics({
        "val/precision": final_metrics.get("metrics/precision", np.nan),
        "val/recall": final_metrics.get("metrics/recall", np.nan),
        "val/f1": 2 * final_metrics.get("metrics/precision", 0) * final_metrics.get("metrics/recall", 0) /
                  (final_metrics.get("metrics/precision", 0) + final_metrics.get("metrics/recall", 0) + 1e-6),
        "val/mAP50": final_metrics.get("metrics/mAP50", np.nan),
        "val/mAP75": final_metrics.get("metrics/mAP75", np.nan),
        "val/IoU@0.5": final_metrics.get("metrics/mAP50", np.nan),   # usually mAP50 ≈ IoU@0.5
        "val/IoU@0.75": final_metrics.get("metrics/mAP75", np.nan),
        "val/accuracy": final_metrics.get("metrics/accuracy", np.nan),
    })
except Exception as e:
    print(f"[WARNING] Final val metric logging failed: {e}")

# ==== OPTIONAL: LOG TEST METRICS ====
try:
    test_results = model.val(split="test")  # Evaluate on test split
    if hasattr(test_results, "metrics"):
        test_metrics = test_results.metrics
        experiment.log_metrics({
            "test/precision": test_metrics.get("metrics/precision", np.nan),
            "test/recall": test_metrics.get("metrics/recall", np.nan),
            "test/f1": 2 * test_metrics.get("metrics/precision", 0) * test_metrics.get("metrics/recall", 0) /
                       (test_metrics.get("metrics/precision", 0) + test_metrics.get("metrics/recall", 0) + 1e-6),
            "test/mAP50": test_metrics.get("metrics/mAP50", np.nan),
            "test/mAP75": test_metrics.get("metrics/mAP75", np.nan),
            "test/IoU@0.5": test_metrics.get("metrics/mAP50", np.nan),
            "test/IoU@0.75": test_metrics.get("metrics/mAP75", np.nan),
            "test/accuracy": test_metrics.get("metrics/accuracy", np.nan),
        })
except Exception as e:
    print(f"[WARNING] Test metric logging failed: {e}")

# ==== LOG MODEL ARTIFACT ====
best_model_path = f"runs/train/{run_name}/weights/best.pt"
if os.path.exists(best_model_path):
    experiment.log_model("best_model", best_model_path)

# ==== END COMET EXPERIMENT ====
experiment.end()


# from comet_ml import Experiment
# from ultralytics import YOLO
# import os
# import numpy as np

# # ==== COMET CONFIGURATION ====
# experiment = Experiment(
#     api_key="hE2ZXmFRrxO1jBzFIcUpDG0xU",
#     project_name="benchmarking",
#     workspace="lakshminarayanan-m"
# )

# # ==== EXPERIMENT INFO ====
# experiment.set_name("yolov11-benchmark")

# # ==== TRAINING CONFIG ====
# model_config = "yolo11l.yaml"
# pretrained_weights = "yolo11l.pt"  # or None if training from scratch
# data_config = "/home/endoai/Desktop/MODEL/YOLOv11l/data.yaml"
# batch_size = 16
# epochs = 30
# workers = 8
# image_size = 640
# run_name = "yolov11l-benchmark"
# save_results = True
# plot_results = True

# # ==== LOG HYPERPARAMETERS TO COMET ====
# experiment.log_parameters({
#     "model": model_config,
#     "data": data_config,
#     "batch": batch_size,
#     "epochs": epochs,
#     "workers": workers,
#     "imgsz": image_size,
#     "run_name": run_name,
#     "pretrained": pretrained_weights
# })

# # ==== LOAD AND TRAIN MODEL ====
# if pretrained_weights:
#     model = YOLO(model_config).load(pretrained_weights)
# else:
#     model = YOLO(model_config)

# results = model.train(
#     data=data_config,
#     batch=batch_size,
#     epochs=epochs,
#     workers=workers,
#     imgsz=image_size,
#     val=True,
#     plots=plot_results,
#     save=save_results,
#     name=run_name
# )

# # ==== LOG TRAINING CURVES (IF AVAILABLE) ====
# train_result_dir = f"runs/detect/{run_name}"
# plot_files = ["results.png", "F1_curve.png", "confusion_matrix.png"]
# for file in plot_files:
#     file_path = os.path.join(train_result_dir, file)
#     if os.path.exists(file_path):
#         experiment.log_image(file_path, step=epochs)

# # ==== FINAL VALIDATION METRICS ====
# try:
#     final_metrics = results.metrics.box.mean_results().cpu().numpy().tolist()
#     precision, recall, mAP50, mAP75 = final_metrics[:4]
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
#     experiment.log_metrics({
#         "val/precision": precision,
#         "val/recall": recall,
#         "val/f1": f1,
#         "val/mAP50": mAP50,
#         "val/mAP75": mAP75,
#         "val/IoU@0.5": mAP50,
#         "val/IoU@0.75": mAP75
#     })
# except Exception as e:
#     print(f"[WARNING] Final val metric logging failed: {e}")

# # ==== TEST METRICS ====
# try:
#     test_results = model.val(data=data_config, split="test")
#     test_metrics = test_results.metrics.box.mean_results().cpu().numpy().tolist()
#     t_precision, t_recall, t_mAP50, t_mAP75 = test_metrics[:4]
#     t_f1 = 2 * t_precision * t_recall / (t_precision + t_recall + 1e-6)

#     experiment.log_metrics({
#         "test/precision": t_precision,
#         "test/recall": t_recall,
#         "test/f1": t_f1,
#         "test/mAP50": t_mAP50,
#         "test/mAP75": t_mAP75,
#         "test/IoU@0.5": t_mAP50,
#         "test/IoU@0.75": t_mAP75
#     })
# except Exception as e:
#     print(f"[WARNING] Test metric logging failed: {e}")

# # ==== LOG MODEL ARTIFACT ====
# best_model_path = os.path.join(train_result_dir, "weights", "best.pt")
# if os.path.exists(best_model_path):
#     experiment.log_model("best_model", best_model_path)

# # ==== END COMET EXPERIMENT ====
# experiment.end()
