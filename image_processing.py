import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO



# Load YOLOv8 model (pre-trained on COCO)
model = YOLO("yolov8s.pt")  # You can use yolov8n.pt (nano), yolov8m.pt (medium), etc.

def process_image(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    results = model(image_path)[0]  # Get the first result

    object_positions = []
    relationships = set()

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    class_names = model.names

    for i, box in enumerate(boxes):
        if confs[i] < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = class_names[int(classes[i])]
        w, h = x2 - x1, y2 - y1
        object_positions.append((label, x1, y1, w, h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Relationship detection
    for i in range(len(object_positions)):
        obj1 = object_positions[i]
        for j in range(i + 1, len(object_positions)):
            obj2 = object_positions[j]
            obj1_label, obj1_x, obj1_y, obj1_w, obj1_h = obj1
            obj2_label, obj2_x, obj2_y, obj2_w, obj2_h = obj2

            if obj1_y < obj2_y:
                relationships.add(f"{obj1_label} is above {obj2_label}")
            elif obj1_y > obj2_y:
                relationships.add(f"{obj1_label} is below {obj2_label}")
            if obj1_x < obj2_x:
                relationships.add(f"{obj1_label} is to the left of {obj2_label}")
            elif obj1_x > obj2_x:
                relationships.add(f"{obj1_label} is to the right of {obj2_label}")

            distance = np.sqrt((obj1_x - obj2_x)**2 + (obj1_y - obj2_y)**2)
            if distance < 100:
                relationships.add(f"{obj1_label} is near {obj2_label}")

    # Convert annotated image to base64
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return json.dumps({
        'relationships': list(relationships),
        'image': img_base64
    })
