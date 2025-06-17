import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="banknote.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Banknote Authentication System")
st.write("Masukkan karakteristik statistik uang kertas untuk mendeteksi keaslian.")

# Form input pengguna
variance = st.number_input("Variance", min_value=-10.0, max_value=10.0, value=-4.8392, format="%.4f")
skewness = st.number_input("Skewness", min_value=-10.0, max_value=15.0, value=6.6755, format="%.4f")
curtosis = st.number_input("Curtosis", min_value=-10.0, max_value=15.0, value=-0.24278, format="%.4f")
entropy = st.number_input("Entropy", min_value=-10.0, max_value=5.0, value=-6.5775, format="%.4f")

if st.button("Prediksi Keaslian Uang"):
    # Preprocessing input
    input_data = np.array([[variance, skewness, curtosis, entropy]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(prediction)
    result = label_encoder.inverse_transform([predicted_class])[0]

    if result == 0:
        st.success("✅ **UANG ASLI** - Banknote ini terdeteksi asli")
    else:
        st.error("❌ **UANG PALSU** - Banknote ini terdeteksi palsu")
    
    # Tampilkan confidence score
    confidence = np.max(prediction) * 100
    st.write(f"Confidence Score: {confidence:.2f}%")