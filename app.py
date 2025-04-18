import os
import cv2 # type: ignore
import json
import base64
import traceback
import numpy as np # type: ignore
from flask import Flask, request, jsonify, render_template # type: ignore
from werkzeug.utils import secure_filename # type: ignore
from ultralytics import YOLO # type: ignore

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}
model = YOLO("yolov8s.pt")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(upload_path)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                result = process_image(upload_path)
            elif filename.lower().endswith(('.mp4', '.avi')):
                result = process_video(upload_path)
            else:
                return jsonify({"error": "Unsupported file type"}), 400

            return jsonify(result), 200

        except Exception as e:
            print("❌ Error during file processing:", e)
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400

def process_image(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    results = model(image_path)[0]

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

    for i in range(len(object_positions)):
        obj1 = object_positions[i]
        for j in range(i + 1, len(object_positions)):
            obj2 = object_positions[j]
            label1, x1, y1, _, _ = obj1
            label2, x2, y2, _, _ = obj2

            if y1 < y2:
                relationships.add(f"{label1} is above {label2}")
            elif y1 > y2:
                relationships.add(f"{label1} is below {label2}")
            if x1 < x2:
                relationships.add(f"{label1} is to the left of {label2}")
            elif x1 > x2:
                relationships.add(f"{label1} is to the right of {label2}")
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < 100:
                relationships.add(f"{label1} is near {label2}")

    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return {
        'relationships': list(relationships),
        'image': img_base64
    }

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

        results = model(frame)[0]
        object_positions = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            object_positions.append((label, x, y, w, h))

            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        for i in range(len(object_positions)):
            obj1 = object_positions[i]
            for j in range(i + 1, len(object_positions)):
                obj2 = object_positions[j]
                label1, x1, y1, _, _ = obj1
                label2, x2, y2, _, _ = obj2

                if y1 < y2:
                    all_relationships.add(f"{label1} is above {label2}")
                elif y1 > y2:
                    all_relationships.add(f"{label1} is below {label2}")
                if x1 < x2:
                    all_relationships.add(f"{label1} is to the left of {label2}")
                elif x1 > x2:
                    all_relationships.add(f"{label1} is to the right of {label2}")

        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_base64_list.append(img_base64)

    cap.release()

    return {
        'relationships': list(all_relationships),
        'video_frames': img_base64_list
    }

if __name__ == '__main__':
    app.run(debug=True)
