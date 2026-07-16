from ultralytics import YOLO
import cv2
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # reads variables from .env file

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Paths
VIDEO_PATH = 'data/raw/sample_traffic.mp4'
WEIGHTS_PATH = 'data/weights/yolov8s.pt'

model = YOLO(WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else 'yolov8s.pt')

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video. Check this path: {VIDEO_PATH}")
    exit()

cv2.namedWindow("Multi-lane Counting - Press 'q' to quit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi-lane Counting - Press 'q' to quit", 960, 540)

# Counting line setup
LINE_Y = 650

# Lane setup
NUM_LANES = 3
ROAD_START_PCT = 0.12  # % of frame width where real road begins (adjust this to move lines)

def get_lane_boundaries(frame_width):
    """Returns the x-pixel boundaries for each lane, skipping the barrier area."""
    road_start = int(frame_width * ROAD_START_PCT)
    road_width = frame_width - road_start
    lane_width = road_width // NUM_LANES
    boundaries = [road_start + i * lane_width for i in range(NUM_LANES + 1)]
    return boundaries  # e.g. [road_start, b1, b2, frame_width]

def get_lane(center_x, boundaries):
    """Given an x-coordinate, return which lane number it falls in."""
    for i in range(len(boundaries) - 1):
        if boundaries[i] <= center_x < boundaries[i + 1]:
            return i + 1
    return NUM_LANES  # fallback for edge case at the very right edge

counted_ids = set()
lane_counts = {1: 0, 2: 0, 3: 0}

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
WINDOW_SECONDS = 10
frames_per_window = int(fps * WINDOW_SECONDS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or frame not found.")
        break

    frame_count += 1
    frame_width = frame.shape[1]  # actual width of THIS frame
    boundaries = get_lane_boundaries(frame_width)

    results = model.track(frame, persist=True, conf=0.15)

    boxes = results[0].boxes
    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        xyxy = boxes.xyxy.tolist()
        classes = boxes.cls.int().tolist()  # class ID for each detected box

        for track_id, box, cls_id in zip(ids, xyxy, classes):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if center_y > LINE_Y and track_id not in counted_ids:
                counted_ids.add(track_id)
                lane = get_lane(center_x, boundaries)
                lane_counts[lane] += 1

                vehicle_type = model.names[cls_id]  # e.g. "car", "bus", "motorcycle"

                # Log this individual vehicle event to the database
                supabase.table("detection_events").insert({
                    "lane_id": lane,
                    "vehicle_id": track_id,
                    "vehicle_type": vehicle_type
                }).execute()

    annotated_frame = results[0].plot()
    cv2.line(annotated_frame, (0, LINE_Y), (frame_width, LINE_Y), (0, 255, 255), 2)

    # Draw lane divider lines (only within the road region, skipping barrier)
    for x in boundaries[1:-1]:  # skip the very first and last (road_start and frame edge)
        cv2.line(annotated_frame, (x, 0), (x, frame.shape[0]), (255, 0, 0), 2)

    y_offset = 50
    for lane_num, count in lane_counts.items():
        cv2.putText(annotated_frame, f"Lane {lane_num}: {count}", (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        y_offset += 40

    cv2.imshow("Multi-lane Counting - Press 'q' to quit", annotated_frame)

    if frame_count % frames_per_window == 0:
        print(f"[Window ending at frame {frame_count}] Lane counts: {lane_counts}")

        # Log the aggregated window snapshot to the database
        supabase.table("lane_counts_window").insert({
            "window_end_frame": frame_count,
            "lane_1_count": lane_counts[1],
            "lane_2_count": lane_counts[2],
            "lane_3_count": lane_counts[3]
        }).execute()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Final lane counts: {lane_counts}")
cap.release()
cv2.destroyAllWindows()