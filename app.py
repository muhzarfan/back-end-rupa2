import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import tensorflow as tf

# Konfigurasi
MODEL_PATH = 'model/furniture_model.h5'
IMAGE_SIZE = (128, 128)  
CLASSES = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def get_confidence_evaluation(confidence_percentage):
    """Evaluate confidence level"""
    if confidence_percentage >= 80:
        return "Sangat Bagus", "🌟🌟🌟🌟🌟"
    elif 71 <= confidence_percentage <= 80:
        return "Bagus", "🌟🌟🌟🌟"
    elif 61 <= confidence_percentage <= 70:
        return "Cukup Bagus", "🌟🌟🌟"
    elif 51 <= confidence_percentage <= 60:
        return "Baik", "🌟🌟"
    elif 41 <= confidence_percentage <= 50:
        return "Buruk", "🌟"
    else:
        return "Sangat Buruk", "❌"

def load_furniture_model():
    """Load model h5"""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def prepare_image(img):
    """Prepare image for prediction"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load model saat startup
st.title("Furniture Quality Checker by ruparupa")
model = load_furniture_model()

if model is not None:
    uploaded_file = st.file_uploader("Upload sebuah gambar mebel:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read and process image
            img = Image.open(io.BytesIO(uploaded_file.read()))
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            processed_img = prepare_image(img)
            
            # Make prediction
            predictions = model.predict(processed_img)
            
            # Get highest probability class
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLASSES[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            confidence_percentage = confidence * 100
            
            # Get evaluation
            evaluation, stars = get_confidence_evaluation(confidence_percentage)
            
            # Display results
            st.success(f"Prediction: {predicted_class}")
            
            # Tampilkan confidence dengan evaluasi
            st.write("---")
            st.write(f"Confidence: {confidence_percentage:.2f}%")
            st.write(f"Evaluasi: {evaluation} {stars}")
            
            # Tampilkan color box sesuai evaluasi
            if confidence_percentage >= 80:
                st.markdown(f'<div style="background-color: #28a745; padding: 10px; border-radius: 5px; color: white;">Tingkat kepercayaan sangat tinggi!</div>', unsafe_allow_html=True)
            elif confidence_percentage < 40:
                st.markdown(f'<div style="background-color: #dc3545; padding: 10px; border-radius: 5px; color: white;">Tingkat kepercayaan sangat rendah!</div>', unsafe_allow_html=True)
            
            # Display probabilities untuk semua kelas
            st.write("---")
            st.write("Detail Probabilities per Class:")
            for class_name, prob in zip(CLASSES, predictions[0]):
                prob_percentage = prob * 100
                eval_text, eval_stars = get_confidence_evaluation(prob_percentage)
                st.write(f"{class_name}: {prob_percentage:.2f}% - {eval_text} {eval_stars}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
else:
    st.error("Model failed to load. Please check the model path and try again.")