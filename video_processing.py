import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO

# Load YOLOv8 model once
model = YOLO("yolov8s.pt")  # or yolov8n.pt, yolov8m.pt, yolov8l.pt

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_relationships = set()
    img_base64_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 != 0:
            continue

        height, width = frame.shape[:2]
        results = model(frame)[0]

        object_positions = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            object_positions.append((label, x, y, w, h))

            # Draw bounding box
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Detect relationships
        for i in range(len(object_positions)):
            obj1 = object_positions[i]
            for j in range(i + 1, len(object_positions)):
                obj2 = object_positions[j]
                obj1_label, obj1_x, obj1_y, _, _ = obj1
                obj2_label, obj2_x, obj2_y, _, _ = obj2

                if obj1_y < obj2_y:
                    all_relationships.add(f"{obj1_label} is above {obj2_label}")
                elif obj1_y > obj2_y:
                    all_relationships.add(f"{obj1_label} is below {obj2_label}")
                if obj1_x < obj2_x:
                    all_relationships.add(f"{obj1_label} is to the left of {obj2_label}")
                elif obj1_x > obj2_x:
                    all_relationships.add(f"{obj1_label} is to the right of {obj2_label}")

        # Encode frame to base64
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_base64_list.append(img_base64)

    cap.release()

    return json.dumps({
        'relationships': list(all_relationships),
        'video_frames': img_base64_list
    })
