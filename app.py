from flask import Flask, render_template, request
import os
import shutil
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RUNS_FOLDER = "static/runs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    # 🔥 DEFAULT: NO RESULT
    result_image = None

    if request.method == "POST":
        file = request.files.get("image")

        # 🔥 IF NO FILE → DO NOTHING
        if not file or file.filename == "":
            return render_template("index.html", result_image=None)

        # 🔥 DELETE OLD RESULTS ONLY AFTER FILE IS SELECTED
        if os.path.exists(RUNS_FOLDER):
            shutil.rmtree(RUNS_FOLDER)

        # save upload
        ext = os.path.splitext(file.filename)[1]
        unique_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ext
        input_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(input_path)

        # YOLO detect
        model.predict(
            source=input_path,
            save=True,
            project=RUNS_FOLDER,
            name="detect"
        )

        detect_folder = os.path.join(RUNS_FOLDER, "detect")
        detected_file = os.listdir(detect_folder)[0]

        result_image = f"runs/detect/{detected_file}"

    return render_template("index.html", result_image=result_image)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)



