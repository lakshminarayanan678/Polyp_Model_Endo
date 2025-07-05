# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("/home/endoai/Desktop/MODEL/YOLOv12l/runs/detect/yolov12-benchmark/weights/best.pt")

# # Run inference on 'bus.jpg' with arguments
# model.predict("/home/endoai/Desktop/PAPER/TEST-CMC/video_20250627_113026.mp4", save=True, imgsz=640, conf=0.75,stream=True)

import cv2
from ultralytics import YOLO
from ultralytics import RTDETR

# Load your trained YOLOv12 model
model = YOLO("/home/endoai/Desktop/MODEL/OLD/yolov11n.pt")

# Path to input video
video_path = "/home/endoai/Desktop/PAPER/TEST-CMC/video_20250627_113026.mp4"
cap = cv2.VideoCapture(video_path)

# Prepare video writer to save output
output_path = "output_annotated.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Run inference frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on the frame
    results = model.predict(frame, stream=True, imgsz=640, conf=0.75, verbose=False)

    for r in results:
        annotated_frame = r.plot()

        # Display the frame
        cv2.imshow("YOLOv12 Inference", annotated_frame)

        # Save the annotated frame to output video
        out.write(annotated_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Annotated video saved to: {output_path}")
