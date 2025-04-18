import os
import traceback  # ✅ Added for detailed error logging
from flask import Flask, request, jsonify, render_template  # type: ignore
from werkzeug.utils import secure_filename  # type: ignore
from image_processing import process_image
from video_processing import process_video

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
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
        file_extension = filename.rsplit('.', 1)[1].lower()
        upload_path = os.path.join("uploads", filename)

        try:
            file.save(upload_path)

            if file_extension in ['png', 'jpg', 'jpeg', 'gif']:
                result = process_image(upload_path)
            elif file_extension in ['mp4', 'avi']:
                result = process_video(upload_path)

            return jsonify(result), 200

        except Exception as e:
            print("❌ Error during file processing:", e)
            traceback.print_exc()  # ✅ Prints full traceback to terminal/logs
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
