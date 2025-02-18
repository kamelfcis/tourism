import os
from flask import Flask, request, Response
import torch
from ultralytics import YOLO
from PIL import Image
import io
import json  

app = Flask(__name__)

# Define model path
MODEL_PATH = "static/weights/best.pt"

# Ensure model directory exists
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

# Load YOLOv8 model (Ensure the model is manually placed in the path)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please upload 'best.pt' manually.")

# Load YOLOv8 model (CPU Mode)
model = YOLO(MODEL_PATH)

# Dictionary storing descriptions for each king/queen
king_descriptions = {
    "Tutankhamun": (
        "Tutankhamun (1332–1323 BCE), of the 18th Dynasty, became pharaoh at a young age. "
        "He is best known for his nearly intact tomb, discovered in 1922."
    ),
    "Ramesses II": (
        "Ramesses II (1279–1213 BCE), also known as Ramesses the Great, was one of Egypt's most powerful pharaohs. "
        "He is famed for his military campaigns and monumental constructions."
    ),
    "Akhenaten": (
        "Akhenaten (1353–1336 BCE), of the 18th Dynasty, is known for his religious revolution. "
        "He promoted monotheism by worshiping Aten, the sun disk."
    ),
    "Thutmose III": (
        "Thutmose III (1479–1425 BCE), of the 18th Dynasty, was an exceptional military leader. "
        "He expanded Egypt’s empire significantly through his campaigns."
    ),
    "Nefertiti": (
        "Nefertiti (c. 1370–1330 BCE), of the 18th Dynasty, was the queen and great royal wife of Akhenaten. "
        "Her famous bust remains a symbol of beauty and power."
    )
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return Response(json.dumps({'error': 'No file uploaded'}), status=400, mimetype='application/json')

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))

    # Run inference on CPU
    results = model(img)  # Runs on CPU as per your requirement

    # Get the predicted class
    pred_class = results[0].probs.top1 if hasattr(results[0], "probs") else None
    class_names = model.names  
    predicted_king = class_names[pred_class] if pred_class is not None else "Unknown"

    # Fetch the description
    description = king_descriptions.get(predicted_king, "Description not available.")

    # Response
    response_data = {
        "prediction": predicted_king,
        "description": description
    }

    return Response(json.dumps(response_data, indent=4), mimetype='application/json')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)
