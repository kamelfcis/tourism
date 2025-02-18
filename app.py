import os
from flask import Flask, request, Response, jsonify
import torch
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Define model path
MODEL_PATH = "static/weights/best.pt"

# Ensure model directory exists
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

# Check if model file exists, otherwise raise an error
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"⚠️ Model file not found at {MODEL_PATH}. Please upload 'best.pt' manually."
    )

# Load YOLOv8 model (Forcing CPU mode)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"🚨 Failed to load YOLO model: {e}")

# Dictionary storing descriptions for each king/queen
king_descriptions = {
    "Tutankhamun": "Tutankhamun (1332–1323 BCE) was a young pharaoh of the 18th Dynasty, "
                   "best known for his nearly intact tomb discovered in 1922.",
    "Ramesses II": "Ramesses II (1279–1213 BCE), also known as Ramesses the Great, was one of Egypt's most powerful pharaohs, "
                   "famed for his military campaigns and monumental constructions.",
    "Akhenaten": "Akhenaten (1353–1336 BCE) led a religious revolution by promoting monotheism with the worship of Aten, the sun disk.",
    "Thutmose III": "Thutmose III (1479–1425 BCE) was an exceptional military leader who significantly expanded Egypt’s empire.",
    "Nefertiti": "Nefertiti (c. 1370–1330 BCE) was the queen and great royal wife of Akhenaten, "
                 "her famous bust remains a symbol of beauty and power.",
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please upload an image file.'}), 400

    file = request.files['file']

    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({'error': 'Invalid image file. Please upload a valid image.'}), 400

    try:
        # Run inference on CPU
        results = model(img, device="cpu")

        # Get the predicted class
        pred_class = results[0].probs.top1 if hasattr(results[0], "probs") else None
        class_names = model.names  
        predicted_king = class_names[pred_class] if pred_class is not None else "Unknown"

        # Fetch the description
        description = king_descriptions.get(predicted_king, "Description not available.")

        # Response
        return jsonify({
            "prediction": predicted_king,
            "description": description
        })

    except Exception as e:
        return jsonify({'error': f'Model inference failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)
