from flask import Flask, render_template, request
import os
import shutil
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model (use best.pt if you have it)
model = YOLO("best.pt")  # change to yolov8n.pt ONLY if best.pt not available

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            return render_template("index.html", result_image=None)

        # 🔥 Clear old detection results safely
        detect_path = os.path.join(RESULTS_FOLDER, "detect")
        if os.path.exists(detect_path):
            shutil.rmtree(detect_path)

        # Save uploaded image
        ext = os.path.splitext(file.filename)[1]
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
        input_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(input_path)

        # Run YOLO detection
        model.predict(
            source=input_path,
            save=True,
            project=RESULTS_FOLDER,
            name="detect"
        )

        # Get detected image safely
        if os.path.exists(detect_path):
            files = os.listdir(detect_path)
            if len(files) > 0:
                detected_file = files[0]
                result_image = f"results/detect/{detected_file}"

    return render_template("index.html", result_image=result_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
