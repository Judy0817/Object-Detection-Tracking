import os
import json
import cv2
import numpy as np
import time
from collections import defaultdict

SOURCE_VIDEO_PATH = "animal/animal detection.mp4"
TARGET_VIDEO_PATH = 'animal_output.mp4'
FRAME_SAVE_DIR = 'animal_frames/'  
FRAME_DATA_PATH = 'animal_data.json'
MODEL_DIR = 'models'

PROTO_TXT = "animal\\models\\MobileNetSSD_deploy.prototxt.txt"
CAFFE_MODEL = "animal\\models\\MobileNetSSD_deploy.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", "dining-table", "dog", 
           "horse", "motorbike", "person", "potted plant", "sheep", 
           "sofa", "train", "monitor"]
REQ_CLASSES = ["bird", "cat", "cow", "dog", "horse", "sheep"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
CONF_THRESH = 0.2


def convert_to_serializable(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_json(data, path):
    """Save data to JSON with proper type conversion"""
    serializable_data = json.loads(json.dumps(data, default=convert_to_serializable))
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=4)

def format_detection_counts(detections):
    """Format detection counts like '1 bird, 2 dogs'"""
    count_dict = defaultdict(int)
    for det in detections:
        count_dict[det['class_name']] += 1
    
    parts = []
    for cls, cnt in count_dict.items():
        if cnt == 1:
            parts.append(f"1 {cls}")
        else:
            parts.append(f"{cnt} {cls}s")
    return ", ".join(parts)


os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load model
net = cv2.dnn.readNetFromCaffe(PROTO_TXT, CAFFE_MODEL)

# Initialize video capture
vs = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not vs.isOpened():
    raise ValueError(f"Error opening video file: {SOURCE_VIDEO_PATH}")

# Get video info
frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vs.get(cv2.CAP_PROP_FPS))
total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize tracking
frame_data_list = []

print(f"Processing video: {SOURCE_VIDEO_PATH}")
print(f"Video dimensions: {frame_width}x{frame_height}")
print(f"Total frames: {total_frames}")


while True:
    start_time = time.time()
    success, frame = vs.read()
    if not success:
        break
        
    frame_number = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp_sec = round(float(vs.get(cv2.CAP_PROP_POS_MSEC)) / 1000, 2)
    
    # Prepare blob and perform detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    detected_animals = []
    current_detections = []
    
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONF_THRESH:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in REQ_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)
                
                # Create detection info with native Python types
                detection_info = {
                    "class_id": int(idx),
                    "class_name": CLASSES[idx],
                    "confidence": float(confidence),
                    "bbox": [int(startX), int(startY), int(endX), int(endY)],
                    "center": {
                        "x": float((startX + endX) / 2),
                        "y": float((startY + endY) / 2)
                    },
                    "area": int((endX - startX) * (endY - startY)),
                    "frame_number": int(frame_number),
                    "timestamp": float(timestamp_sec)
                }
                detected_animals.append(detection_info)
                current_detections.append(CLASSES[idx])
                
                # Draw on frame
                label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Print frame info in the requested format
    detection_str = format_detection_counts(detected_animals) if detected_animals else "No animals detected"
    print(f"video 1/1 (frame {frame_number}/{total_frames}) {SOURCE_VIDEO_PATH}: {frame_width}x{frame_height} {detection_str}, {processing_time:.1f}ms")
    
    # Save frame data
    if detected_animals:
        frame_data_list.append({
            "frame_number": int(frame_number),
            "timestamp": float(timestamp_sec),
            "detections": detected_animals
        })
        
        # Save frame image
        frame_path = os.path.join(FRAME_SAVE_DIR, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_path, frame)

vs.release()

# Save JSON data with proper type conversion
save_json(frame_data_list, FRAME_DATA_PATH)

print("\nAnimal detection completed. Results saved to:")
print(f"- Frames directory: {FRAME_SAVE_DIR}")
print(f"- Detection data: {FRAME_DATA_PATH}")