
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "model/nature_classification_final.keras"
CLASS_NAMES_PATH = "model/class_names.json"
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Nature Classification",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------
# Load model and class names
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)

# -----------------------------
# Application UI
# -----------------------------
st.title("🌿 Nature Classification")
st.write(
    "Upload an image of an animal or plant "
    "to identify its category using our trained AI model."
)

try:
    model = load_model()
    class_names = load_class_names()
    st.success("AI model loaded successfully!")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize(IMAGE_SIZE)

            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)

            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index]) * 100

            predicted_class = class_names[predicted_index]

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {predicted_class.replace('_', ' ').title()}")
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Top 5 Predictions")

        top_indices = np.argsort(predictions[0])[-5:][::-1]

        for index in top_indices:
            class_name = class_names[index].replace("_", " ").title()
            score = float(predictions[0][index]) * 100

            st.write(f"**{class_name}** — {score:.2f}%")
            st.progress(min(score / 100, 1.0))
