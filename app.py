from flask import Flask, render_template, request
import os
import shutil
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RUNS_FOLDER = "static/runs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None

    if request.method == "POST":
        file = request.files.get("image")

        # If no file selected
        if not file or file.filename == "":
            return render_template("index.html", result_image=None)

        # Remove old detection results
        if os.path.exists(RUNS_FOLDER):
            shutil.rmtree(RUNS_FOLDER)

        # Save uploaded image
        ext = os.path.splitext(file.filename)[1]
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
        input_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(input_path)

        # Run YOLO detection
        model.predict(
            source=input_path,
            save=True,
            project=RUNS_FOLDER,
            name="detect"
        )

        detect_folder = os.path.join(RUNS_FOLDER, "detect")

        # Safety check (VERY IMPORTANT)
        if os.path.exists(detect_folder) and os.listdir(detect_folder):
            detected_file = os.listdir(detect_folder)[0]
            result_image = f"runs/detect/{detected_file}"

    return render_template("index.html", result_image=result_image)

if __name__ == "__main__":
    app.run(debug=False)


