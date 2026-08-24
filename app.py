# Copyright (C) 2026 Your Name / Company Name
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

import base64
import io
from flask import Flask, jsonify, render_template, request
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load model
MODEL_PATH = "train-4.torchscript"  # Update to your actual filename (.onnx, .torchscript, or .pt)
model = YOLO(MODEL_PATH)

print("\n--- MODEL LOADED SUCCESSFULLY ---")
print("Detected Classes in Model:", getattr(model, 'names', 'No class names found'))
print("---------------------------------\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # 1. Clean Base64 String
        raw_image_str = data["image"]
        if "," in raw_image_str:
            raw_image_str = raw_image_str.split(",")[1]

        image_bytes = base64.b64decode(raw_image_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Run Inference with low confidence (5%)
        results = model(image, conf=0.01, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id] if hasattr(model, 'names') and cls_id in model.names else f"class_{cls_id}"
                xyxy = box.xyxy[0].tolist()

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "box": xyxy
                })

        # 3. Terminal Diagnostics
        if detections:
            print(f"✅ DETECTED ({len(detections)}):", [(d['class'], d['confidence']) for d in detections])
        else:
            print("❌ Frame processed: No objects detected above 5% confidence.")

        return jsonify({"detections": detections})

    except Exception as e:
        print("⚠️ ERROR processing frame:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)