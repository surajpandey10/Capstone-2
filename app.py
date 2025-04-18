import os
from flask import Flask, request, jsonify, render_template # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import json
from image_processing import process_image  # Updated to use YOLOv8
from video_processing import process_video  # You should also update this similarly

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# Check if file is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        upload_path = os.path.join("uploads", filename)

        try:
            # Save file
            file.save(upload_path)

            # Image
            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                result = process_image(upload_path)

            # Video
            elif file_extension in ['mp4', 'avi']:
                result = process_video(upload_path)  # Make sure this is YOLOv8-compatible too

            return jsonify(json.loads(result)), 200

        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    return jsonify({"error": "File type not allowed"}), 400

# Handle favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 204

# Ensure 'uploads' folder exists
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
