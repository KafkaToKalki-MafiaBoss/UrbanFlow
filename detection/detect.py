from ultralytics import YOLO
import cv2
import os

# --- Paths (based on your folder structure) ---
VIDEO_PATH = 'data/raw/sample_traffic.mp4'
WEIGHTS_PATH = 'data/weights/yolov8n.pt'

# If the model isn't already in the weights folder, this will auto-download it
model = YOLO(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else 'yolov8n.pt')

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video. Check this path: {VIDEO_PATH}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()   # grab one frame from the video
    if not ret:
        print("Video ended or frame not found.")
        break

    # Run YOLO on this frame — this is the actual detection step
    results = model(frame)

    # Draw boxes + labels + confidence on the frame (built-in helper)
    annotated_frame = results[0].plot()

    # Show it on screen
    cv2.imshow("Detection - Press 'q' to quit", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()