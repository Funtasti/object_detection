import os
import csv
from ultralytics import YOLO
import cv2
from utils import extract_frames, save_video

VIDEO_PATH = 'input_video.mp4'
OUTPUT_VIDEO = 'output/annotated_video.mp4'
CSV_FILE = 'output/detections.csv'
MODEL_NAME = 'yolov8n.pt'

# Create output directory
os.makedirs('output', exist_ok=True)

model = YOLO(MODEL_NAME)

detection_report = []
annotated_frames = []

# Video properties
video = cv2.VideoCapture(VIDEO_PATH)
fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
video.release()

for frame_index, frame in extract_frames(VIDEO_PATH):
    result = model(frame)
    for r in result:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            detection_report.append({
                'frame' : frame_index,
                'label' : label,
                'confidence' : conf,
                'x1' : xyxy[0], 'y1' : xyxy[1], 'x2' : xyxy[2], 'y2' : xyxy[3]
            })
            # Annotate frame with label and meaning
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255, 0), 2)
            cv2.putText(frame, f"{label} {conf: .2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    annotated_frames.append(frame)

save_video(annotated_frames, fps, (width, height), OUTPUT_VIDEO)

#CSV Report
keys = ['frame', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2']
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(detection_report)

print(f"Annotated video saved to {OUTPUT_VIDEO}")
print(f"Detections report saved to {CSV_FILE}")