import os
from flask import Flask, request, Response
from flask_cors import CORS
import torch
from ultralytics import YOLO
from PIL import Image
import io
import json  

app = Flask(__name__)
CORS(app) 

# Load the trained YOLOv8 model 
MODEL_PATH = os.path.join(os.getcwd(), "static", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# Dictionary storing descriptions for each king/queen
king_descriptions = {
    "Tutankhamun": (
        "Tutankhamun (1332–1323 BCE), of the 18th Dynasty, became pharaoh at a young age. "
        "He is best known for his nearly intact tomb, discovered in 1922, which provided an unprecedented insight into ancient Egyptian burial customs. "
        "His reign marked the restoration of traditional Egyptian religion after the radical changes introduced by his predecessor, Akhenaten."
    ),
    "Ramesses II": (
        "Ramesses II (1279–1213 BCE), also known as Ramesses the Great, was one of Egypt's most powerful and longest-reigning pharaohs, ruling during the 19th Dynasty. "
        "He is famed for his military campaigns, the construction of the magnificent Abu Simbel temples, and signing the first recorded peace treaty with the Hittites. "
        "His legacy is marked by grand monuments and an era of prosperity."
    ),
    "Akhenaten": (
        "Akhenaten (1353–1336 BCE), of the 18th Dynasty, is often called the 'heretic pharaoh' for his religious revolution. "
        "He abandoned the traditional Egyptian pantheon in favor of worshiping Aten, the sun disk, establishing Egypt's first monotheistic religion. "
        "He built the city of Amarna as the new religious capital, but his reforms were reversed after his death."
    ),
    "Thutmose III": (
        "Thutmose III (1479–1425 BCE), of the 18th Dynasty, is known as the 'Napoleon of Egypt' due to his unparalleled military campaigns. "
        "He expanded Egypt’s empire to its greatest territorial extent, leading 17 victorious military expeditions. "
        "His reign also saw advancements in art, architecture, and administration, securing Egypt's golden age."
    ),
    "Nefertiti": (
        "Nefertiti (c. 1370–1330 BCE), of the 18th Dynasty, was the queen and great royal wife of Akhenaten. "
        "She played a crucial role in the religious transformation of Egypt, alongside her husband, promoting the worship of Aten. "
        "Her iconic bust, discovered in 1912, remains one of the most famous symbols of ancient Egyptian beauty and power."
    )
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return Response(json.dumps({'error': 'No file uploaded'}), status=400, mimetype='application/json')
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    
    # Run inference
    results = model(img)
    
    # Get the predicted class
    #pred_class = results[0].probs.top1 
    pred_class = results[0].probs.top1 if hasattr(results[0], "probs") else None 
    class_names = model.names  
    predicted_king = class_names[pred_class]

    # Fetch the description
    description = king_descriptions.get(predicted_king, "Description not available.")

    # oreder the response 
    response_data = {
        "prediction": predicted_king,
        "description": description
    }

    return Response(json.dumps(response_data, indent=4), mimetype='application/json')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)
