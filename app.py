import json
import os

import numpy as np
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "nature_classification_final.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
SPECIES_INFO_PATH = os.path.join(BASE_DIR, "data", "species_info.json")
IMAGE_SIZE = (224, 224)
LOW_CONFIDENCE = 50.0

st.set_page_config(page_title="Nature Classification", page_icon="🌿", layout="centered")


# -----------------------------
# Loaders
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as file:
        return json.load(file)


def load_species_info():
    # Not cached, so a missing file is reported and fixed without clearing cache
    try:
        with open(SPECIES_INFO_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        st.warning(f"species_info.json not found at: {SPECIES_INFO_PATH}")
        return {}
    except json.JSONDecodeError as error:
        st.warning(f"species_info.json has a formatting error: {error}")
        return {}


@st.cache_data(show_spinner=False)
def fetch_wikipedia(title):
    """Fallback: short summary from Wikipedia for classes without local info."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
        response = requests.get(
            url, timeout=8, headers={"User-Agent": "NatureClassificationApp/1.0"}
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "extract": data.get("extract", ""),
                "image": data.get("thumbnail", {}).get("source"),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
            }
    except requests.RequestException:
        pass
    return None


def wiki_link(title):
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


# -----------------------------
# Education page
# -----------------------------
def show_education_page(class_key, species_info):
    pretty_name = class_key.replace("_", " ").title()
    info = species_info.get(class_key)

    st.divider()

    if info:
        st.header(f"📚 {info['common_name']}")
        st.caption(f"*{info['scientific_name']}*")

        # Quick facts
        col1, col2, col3 = st.columns(3)
        col1.metric("Type", info["type"])
        col2.metric("Wild / Cultivated", info["wild_status"])
        col3.metric("Scientific name", info["scientific_name"])

        # Conservation status banner
        status_text = f"Conservation status: {info['conservation_status']}"
        level = info.get("status_level", "unknown")
        if level == "threatened":
            st.error("⚠️ " + status_text)
        elif level == "safe":
            st.success("✅ " + status_text)
        else:
            st.info("ℹ️ " + status_text)

        # Brief information
        st.subheader("About")
        st.write(info["brief"])

        # Detail tabs
        tab1, tab2, tab3 = st.tabs(["Habitat & Food/Uses", "Threats", "Fun Facts"])
        with tab1:
            st.markdown(f"**🌍 Habitat:** {info['habitat']}")
            if "diet" in info:
                st.markdown(f"**🍃 Diet:** {info['diet']}")
            if "uses" in info:
                st.markdown(f"**🌿 Uses & benefits:** {info['uses']}")
        with tab2:
            st.write(info["threats"])
        with tab3:
            for fact in info["fun_facts"]:
                st.markdown(f"- {fact}")

        st.markdown(
            f"[Read more on Wikipedia]({wiki_link(info.get('wikipedia', pretty_name))})"
        )
    else:
        st.header(f"📚 Learn about: {pretty_name}")
        wiki = fetch_wikipedia(pretty_name)
        if wiki and wiki["extract"]:
            if wiki["image"]:
                st.image(wiki["image"], width=250)
            st.write(wiki["extract"])
            if wiki["url"]:
                st.markdown(f"[Read more on Wikipedia]({wiki['url']})")
        else:
            st.warning("No information available yet. Add it to data/species_info.json.")


# -----------------------------
# Application UI
# -----------------------------
st.title("🌿 Nature Classification")
st.write(
    "Upload an image of an animal or plant to identify it, "
    "then learn about it on its education page."
)

try:
    model = load_model()
    class_names = load_class_names()
    species_info = load_species_info()
    st.success("AI model loaded successfully!")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)
            predictions = model.predict(image_array, verbose=0)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index]) * 100

        # Keep results so the page stays visible when the app reruns
        st.session_state["result"] = {
            "predictions": predictions[0].tolist(),
            "index": predicted_index,
            "confidence": confidence,
            "class": class_names[predicted_index],
        }

result = st.session_state.get("result")
if uploaded_file is not None and result:
    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {result['class'].replace('_', ' ').title()}")
    st.metric("Confidence", f"{result['confidence']:.2f}%")

    if result["confidence"] < LOW_CONFIDENCE:
        st.warning(
            "The model is not very sure about this image. "
            "Try a clearer photo of a single animal or plant."
        )

    st.subheader("Top 5 Predictions")
    scores = np.array(result["predictions"])
    for index in np.argsort(scores)[-5:][::-1]:
        name = class_names[index].replace("_", " ").title()
        score = float(scores[index]) * 100
        st.write(f"**{name}** — {score:.2f}%")
        st.progress(min(score / 100, 1.0))

    show_education_page(result["class"], species_info)