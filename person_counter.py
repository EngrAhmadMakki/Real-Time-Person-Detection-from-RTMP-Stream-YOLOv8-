import cv2
import time
import os
from ultralytics import YOLO

# Create results folder
os.makedirs("results", exist_ok=True)

# Load YOLOv8 nano model
model = YOLO('yolov8n.pt')

# RTMP stream URL
stream_url = 'rtmp://vtmsgpzl.ezvizlife.com:1935/v3/openlive/K57001616_1_1?expire=1814948471&id=865991537063763968&c=9ada396462&t=bfbd6a37b49ddb6af168aa76923667d767604ecae975008da995d5a82d1f653b&ev=100'
cap = cv2.VideoCapture(stream_url)

# Wait for stream to open
if not cap.isOpened():
    print("❌ Cannot open stream.")
    exit()

# Get width and height from stream
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = 10  # Set FPS manually since RTMP might not report correctly

# Create VideoWriter object
video_filename = 'results/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

# Time control
start_time = time.time()
duration = 20  # seconds

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    results = model(frame)
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # Person class
                person_count += 1
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

    # Overlay text
    cv2.putText(frame, f'Persons: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write frame to video
    out.write(frame)

    # Optional: show preview
    # cv2.imshow('Preview', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Saved 20-second video to {video_filename}")
