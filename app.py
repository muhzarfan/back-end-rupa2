from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# kelas klasifikasi model
CLASSES = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

# load model
def load_model():
    try:
        model = tf.keras.models.load_model('model/furniture_model.keras')
        print("Model berhasil dimuat!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

model = load_model()

def preprocess_image(image):
    try:
        # konversi ke RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # ubah ukuran gambar
        image = image.resize((224, 224))
        
        # konversi ke array
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        
        # normalisasi gambar
        image_array = image_array / 255.0
        
        # dimensi batch
        image_array = np.expand_dims(image_array, axis=0)
        
        print(f"Input shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    try:
        # baca file gambar
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        
        # preproses gambar
        processed_image = preprocess_image(image)
        
        # menampilkan shape sebelum prediksi
        print(f"Final input shape: {processed_image.shape}")
        
        # prediksi
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASSES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # hasil prediksi dalam bentuk json
        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'input_shape': processed_image.shape
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': str(type(e).__name__)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)