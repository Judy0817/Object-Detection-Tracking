# convert_json_to_yolo.py
import json
import os
import cv2

annotation_dir = 'dataset/annotations'
yolo_dir = 'dataset/labels'
os.makedirs(yolo_dir, exist_ok=True)
class_map = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'motorcycle': 3,
    'bicycle': 4,
    'tuk-tuk': 5  # Add tuk-tuk class
}

for json_file in os.listdir(annotation_dir):
    if not json_file.endswith('.json'):
        continue
    with open(os.path.join(annotation_dir, json_file), 'r') as f:
        annotations = json.load(f)
    image_name = json_file.replace('.json', '.png')
    image_path = os.path.join('dataset/images', image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f'Failed to load {image_name}')
        continue
    img_height, img_width = image.shape[:2]
    yolo_lines = []
    for ann in annotations:
        if ann['class'] not in class_map:
            continue
        bbox = ann['bbox']
        x_center = (bbox[0] + bbox[2]) / 2 / img_width
        y_center = (bbox[1] + bbox[3]) / 2 / img_height
        width = (bbox[2] - bbox[0]) / img_width
        height = (bbox[3] - bbox[1]) / img_height
        class_id = class_map[ann['class']]
        yolo_lines.append(f'{class_id} {x_center} {y_center} {width} {height}')
    with open(os.path.join(yolo_dir, json_file.replace('.json', '.txt')), 'w') as f:
        f.write('\n'.join(yolo_lines))
    print(f'Converted {json_file} to YOLO format')

    classes = ['tuk-tuk', 'motorcycle', 'car', 'van',  'bicycle', 'truck', 'bus']
    with open('dataset/classes.txt', 'w') as f:
        f.write('\n'.join(classes))
