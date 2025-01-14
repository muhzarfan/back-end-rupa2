from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Konfigurasi
MODEL_PATH = 'model/furniture_model.h5'
IMAGE_SIZE = (128, 128)  # Ubah sesuai dengan training model Anda
CLASSES = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def load_furniture_model():
    """Load model h5"""
    try:
        model = load_model(MODEL_PATH)
        print("Model berhasil dimuat")
        # Print model architecture
        model.summary()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_image(img):
    """Prepare image for prediction"""
    # Convert ke RGB jika perlu
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(IMAGE_SIZE)
    
    # Convert ke array
    img_array = image.img_to_array(img)
    
    # Normalisasi
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load model saat startup
print("Loading model...")
model = load_furniture_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        # Get image file
        file = request.files['image']
        
        # Read and process image
        img = Image.open(io.BytesIO(file.read()))
        processed_img = prepare_image(img)
        
        # Make prediction
        predictions = model.predict(processed_img)
        
        # Get highest probability class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Return result
        result = {
            'status': 'success',
            'class': predicted_class,
            'confidence': f'{confidence * 100:.2f}%',
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(CLASSES, predictions[0])
            }
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health():
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'message': 'Service is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)