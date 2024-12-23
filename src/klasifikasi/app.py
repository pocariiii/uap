import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from PIL import Image
import altair as alt

# Dimensi gambar
img_width, img_height = 128, 128

# Load model
@st.cache_resource
def load_mobilenet_model():
    return load_model('src/klasifikasi/model/mobilenet_model.h5')

model = load_mobilenet_model()

# Class labels
class_labels = ['Organik', 'Non-Organik']

# Fungsi untuk prediksi
def predict_image(image):
    # Resize gambar
    image = image.resize((img_width, img_height))
    image_array = img_to_array(image) / 255.0  # Normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_labels[predicted_class], confidence, predictions[0]

# Header
st.markdown(
    """
    <div style="text-align: center;">
        <h1>🗑️ Klasifikasi Sampah</h1>
        <p style="font-size: 18px; color: gray;">
            Aplikasi ini membantu mengklasifikasikan jenis sampah menjadi 2 kategori utama:<br>
            <b>Organik:</b> Sampah yang dapat terurai secara alami<br>
            <b>Non-Organik:</b> Sampah yang tidak dapat terurai secara alami
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Divider
st.markdown("---")

# Layout untuk input gambar
col1, col2 = st.columns(2, gap="large")

# Input gambar melalui upload
with col1:
    st.markdown("### 📤 Upload Gambar")
    uploaded_files = st.file_uploader(
        "Unggah satu atau lebih gambar sampah",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

# Input gambar melalui kamera
with col2:
    st.markdown("### 📷 Ambil Gambar dengan Kamera")
    camera_images = st.camera_input("Gunakan kamera untuk mengambil gambar")

# Divider untuk hasil prediksi
st.markdown("---")

images_to_process = []

# Proses gambar yang diupload
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        images_to_process.append(image)

# Proses gambar dari kamera
if camera_images:
    camera_image = Image.open(camera_images)
    images_to_process.append(camera_image)

# Hasil Prediksi
if images_to_process:
    st.markdown("### 📝 Hasil Prediksi")
    for i, image in enumerate(images_to_process):
        col1, col2 = st.columns([1, 2], gap="large")

        # Tampilkan gambar
        with col1:
            st.image(image, caption=f"Gambar {i + 1}", use_container_width=True)

        # Prediksi
        predicted_class, confidence, probabilities = predict_image(image)

        # Tampilkan hasil prediksi
        with col2:
            st.markdown(
                f"""
                <div style="font-size: 18px;">
                    <b>Prediksi Gambar {i + 1}:</b><br>
                    - Kategori : {predicted_class}<br>
                    - Probabilitas : {confidence:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Visualisasi probabilitas dengan Altair
        st.markdown("### 📊 Visualisasi Probabilitas")
        probabilities_df = pd.DataFrame({
            "Kelas": class_labels,
            "Probabilitas": probabilities
        })

        chart = (
            alt.Chart(probabilities_df)
            .mark_bar()
            .encode(
                x=alt.X("Kelas", sort="-y", title="Kategori Sampah"),
                y=alt.Y("Probabilitas", title="Probabilitas (%)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Kelas", scale=alt.Scale(scheme="tableau10")),
            )
            + alt.Chart(probabilities_df)
            .mark_text(dy=-10)
            .encode(
                x=alt.X("Kelas", sort="-y"),
                y=alt.Y("Probabilitas"),
                text=alt.Text("Probabilitas:Q", format=".2%"),
            )
        ).properties(width=600, height=400)
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("Silakan unggah gambar atau gunakan kamera untuk mengambil gambar.")
