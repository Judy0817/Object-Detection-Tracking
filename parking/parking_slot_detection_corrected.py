import cv2
import numpy as np
import pickle
import json
from datetime import datetime

# ===== Configuration =====
VIDEO_PATH = "parking/parking_crop.mp4"
SLOT_COORDS_PATH = "parking/outputs/parking_slot_coords.pkl"
OUTPUT_VIDEO_PATH = "parking/outputs/parking_slot_output_with_boxes.mp4"
OUTPUT_JSON_PATH = "parking/outputs/parking_detection_output.json"
THRESHOLD = 900

# ===== Load Parking Slot Coordinates =====
with open(SLOT_COORDS_PATH, 'rb') as f:
    parking_slots = pickle.load(f)

# ===== Open Video =====
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(" Cannot open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# ===== Helper Function =====
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
    return dilated

# ===== Tracking State for JSON =====
frame_id = 0
slot_status = {str(i+1): [] for i in range(len(parking_slots))}
occupancy_start_time = {}
vehicle_data = {}
current_status = {}

# ===== Process Video Frame-by-Frame =====
while True:
    success, frame = cap.read()
    if not success:
        break

    frame_time_sec = frame_id / fps
    processed = preprocess_frame(frame)
    free_count = 0
    final_occupancy = {}

    for idx, (x, y, w, h) in enumerate(parking_slots):
        slot_id = str(idx + 1)
        crop = processed[y:y+h, x:x+w]
        count = cv2.countNonZero(crop)
        occupied = count >= THRESHOLD
        color = (0, 0, 255) if occupied else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, slot_id, (x+2, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        final_occupancy[slot_id] = occupied
        slot_status[slot_id].append({
            "frame": frame_id,
            "time": round(frame_time_sec, 2),
            "occupied": occupied
        })

        # track entry/exit time
        if slot_id not in current_status:
            current_status[slot_id] = occupied
            if occupied:
                occupancy_start_time[slot_id] = frame_time_sec
        elif current_status[slot_id] != occupied:
            if occupied:
                occupancy_start_time[slot_id] = frame_time_sec
            else:
                start = occupancy_start_time.get(slot_id, None)
                if start is not None:
                    duration = round(frame_time_sec - start, 2)
                    vehicle_data.setdefault(slot_id, []).append({
                        "entry_time": round(start, 2),
                        "exit_time": round(frame_time_sec, 2),
                        "duration_sec": duration
                    })
                    del occupancy_start_time[slot_id]
            current_status[slot_id] = occupied

        if not occupied:
            free_count += 1

    cv2.putText(frame, f"Free: {free_count}/{len(parking_slots)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    out.write(frame)
    frame_id += 1

# ===== Finalize =====
# Close any unclosed occupancy
end_time = total_frames / fps
for slot_id, start in occupancy_start_time.items():
    vehicle_data.setdefault(slot_id, []).append({
        "entry_time": round(start, 2),
        "exit_time": round(end_time, 2),
        "duration_sec": round(end_time - start, 2)
    })

cap.release()
out.release()

# ===== Save JSON =====
output_json = {
    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "video_source": VIDEO_PATH.split("/")[-1],
    "video_info": {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames
    },
    "parking_config": {
        "total_slots": len(parking_slots),
        "slot_coordinates": {
            str(i+1): [ [parking_slots[i][0], parking_slots[i][1]],
                        [parking_slots[i][0]+parking_slots[i][2], parking_slots[i][1]],
                        [parking_slots[i][0]+parking_slots[i][2], parking_slots[i][1]+parking_slots[i][3]],
                        [parking_slots[i][0], parking_slots[i][1]+parking_slots[i][3]] ]
            for i in range(len(parking_slots))
        }
    },
    "results": {
        "total_vehicles": sum(len(v) for v in vehicle_data.values()),
        "final_occupancy": {k: v[-1]["occupied"] if v else False for k, v in slot_status.items()},
        "vehicle_data": vehicle_data
    }
}

with open(OUTPUT_JSON_PATH, "w") as jf:
    json.dump(output_json, jf, indent=4)

print(" Video and JSON saved.")
