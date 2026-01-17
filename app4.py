from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allows frontend to access this API (handles CORS)

# Load your trained model (adjust filename if needed)
model = tf.keras.models.load_model("skin_disease_model.h5")

# Define disease names and corresponding cures
diseases = [
    "Eczema",
    "Psoriasis",
    "Melanoma",
    "Acne",
    "Rosacea",
    "Vitiligo",
    "Warts",
    "Ringworm",
    "Dermatitis",
    "Lupus"
]

cures = {
    "Eczema": "Use moisturizers, avoid triggers, and apply corticosteroid creams.",
    "Psoriasis": "Use medicated creams, phototherapy, or immunosuppressants.",
    "Melanoma": "Consult oncologist immediately. Requires surgical removal and therapy.",
    "Acne": "Use salicylic acid, benzoyl peroxide or consult dermatologist for retinoids.",
    "Rosacea": "Avoid triggers. Use antibiotics or prescription gels.",
    "Vitiligo": "Use corticosteroids, UV therapy or camouflage makeup.",
    "Warts": "Apply salicylic acid, cryotherapy, or laser treatment.",
    "Ringworm": "Use antifungal creams like clotrimazole or terbinafine.",
    "Dermatitis": "Avoid irritants. Use anti-inflammatory creams.",
    "Lupus": "Requires immunosuppressants. Consult a rheumatologist."
}

# Image preprocessing function (adapt according to your model)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # Change if your model expects different size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        img = preprocess_image(image_bytes)
        prediction = model.predict(img)[0]
        class_index = int(np.argmax(prediction))
        confidence = float(prediction[class_index])
        disease = diseases[class_index]
        cure = cures[disease]

        return jsonify({
            "disease": disease,
            "confidence": confidence,
            "cure": cure
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
