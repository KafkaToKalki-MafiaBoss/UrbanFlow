from ultralytics import YOLO
import cv2
import os

# --- Paths ---
VIDEO_PATH = 'data/raw/sample_traffic.mp4'
WEIGHTS_PATH = 'data/weights/yolov8n.pt'

model = YOLO(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else 'yolov8n.pt')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video. Check this path: {VIDEO_PATH}")
    exit()

# --- Counting line setup ---
LINE_Y = 500  # horizontal line position (pixels from top) — adjust after seeing your video

counted_ids = set()   # keeps track of vehicle IDs already counted
vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or frame not found.")
        break

    results = model.track(frame, persist=True)

    # Draw the counting line on the frame so we can see it
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 255, 255), 2)

    boxes = results[0].boxes
    if boxes.id is not None:  # only proceed if tracking IDs exist for this frame
        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()

        for track_id, box in zip(ids, xyxy):
            x1, y1, x2, y2 = box
            center_y = (y1 + y2) / 2  # vertical center of the box

            # If this vehicle's center has passed the line and hasn't been counted yet
            if center_y > LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                vehicle_count += 1

    # Draw the running count on screen
    cv2.putText(frame, f"Count: {vehicle_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    annotated_frame = results[0].plot()
    # Re-draw the line and count on the annotated frame too
    cv2.line(annotated_frame, (0, LINE_Y), (annotated_frame.shape[1], LINE_Y), (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Count: {vehicle_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Detection + Tracking + Counting - Press 'q' to quit", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Final count: {vehicle_count}")
cap.release()
cv2.destroyAllWindows()