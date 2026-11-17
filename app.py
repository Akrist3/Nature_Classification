import html
import json
import os
import time

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
TIMING_SAMPLE_LIMIT = 20

st.set_page_config(
    page_title="Nature Classification",
    page_icon="🌿",
    layout="centered",
)

# -----------------------------
# Styling + motion
# -----------------------------
# Motion rules: motion only answers an action (classify, hover on a result row).
# Nothing animates on its own, and prefers-reduced-motion switches it all off.
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Source+Sans+3:wght@400;600&display=swap');

:root {
  --ink: #17231b;
  --muted: #3a4a3e;
  --accent: #27493a;
  --soft: #dfe8d8;
  --card: #f6f8f2;
  --line: #98ac8e;
  --bg: #c5d2bc;
  --warn: #8a5a0b;
}

html, body, [class*="css"], .stApp {
  font-family: 'Source Sans 3', system-ui, sans-serif;
  font-size: 17px;
  color: var(--ink);
}
.stApp { background: var(--bg); }
[data-testid="stHeader"] { background: var(--bg); }
[data-testid="stHeader"] * { color: var(--ink); }
.stMarkdown, .stMarkdown p, .stMarkdown li, [data-testid="stWidgetLabel"] p { color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted); }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--ink); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--muted); }
[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p { color: var(--ink); }
[data-testid="stFileUploader"] button { background: var(--card); color: var(--ink); border: 1px solid #5f7a67; }
[data-testid="stExpander"] { background: var(--card); border-color: var(--line); border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] p { color: var(--ink); }
.stTabs [data-baseweb="tab"] p { color: inherit; }
[data-baseweb="tab-panel"] p, [data-baseweb="tab-panel"] span, [data-baseweb="tab-panel"] label,
[data-baseweb="tab-panel"] li, [data-testid="stMarkdownContainer"] p { color: var(--ink); }
[data-testid="stAlert"], [data-testid="stAlert"] p, [data-testid="stAlert"] * { color: var(--ink) !important; }
.stButton > button[kind="secondary"] { background: var(--card); border: 1px solid #5f7a67; border-radius: 10px; }
.stButton > button[kind="secondary"], .stButton > button[kind="secondary"] p { color: var(--ink) !important; }
.stButton > button[kind="secondary"]:hover { border-color: var(--ink); background: var(--soft); }
[data-testid="stExpander"] *, [data-testid="stExpander"] summary p { color: var(--ink); }

/* Run-time chart (plain HTML, so it never follows the browser theme) */
.runs { display: flex; align-items: flex-end; gap: 6px; height: 180px; padding: 1rem 1rem .4rem;
  background: var(--card); border: 1px solid var(--line); border-radius: 12px; margin: 0 0 1rem; }
.run { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; min-width: 0; }
.run .col { width: 100%; max-width: 34px; background: var(--accent); border-radius: 6px 6px 0 0; height: var(--h);
  transform-origin: bottom; animation: rise .6s ease-out both; animation-delay: var(--d); }
.run .val { font-size: .72rem; color: var(--ink); margin-bottom: 2px; white-space: nowrap; }
.run .n { font-size: .75rem; color: var(--muted); margin-top: 4px; }
@keyframes rise { from { transform: scaleY(0); } to { transform: scaleY(1); } }
.block-container { max-width: 980px; padding-top: 2.5rem; padding-bottom: 5rem; }
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"], [data-testid="stHeading"] *,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
.page-title, .result-card .name, .ring span {
  font-family: 'Bricolage Grotesque', system-ui, sans-serif; letter-spacing: -0.01em; color: #0a0a0a !important;
}
h4 { font-size: 1.15rem; margin-bottom: .6rem; }
p, li { line-height: 1.65; }

/* Page header */
.page-title { font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.4rem; font-weight: 800; margin: 0; }
.page-sub { color: var(--muted); font-size: 1.1rem; margin: .3rem 0 1.4rem; max-width: 36rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #7f9475; }
.stTabs [data-baseweb="tab"] { padding: 0 .2rem; height: 2.8rem; font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] p { color: var(--ink); }
.stTabs [aria-selected="true"] { color: var(--accent); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); height: 3px; }

/* Panels (one radius, one border, no shadows) */
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--card); border-color: var(--line) !important; border-radius: 12px; }
[data-testid="stFileUploader"] section { border: 1px dashed #5f7a67; background: var(--card); border-radius: 10px; }
[data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--muted); }
.stButton > button[kind="primary"] { background: var(--accent); border: 0; border-radius: 10px; font-weight: 600; padding: .6rem 1.2rem; }
.stButton > button[kind="primary"]:hover { background: var(--ink); }

/* Result card: fades in after each classification (a/b names let it replay) */
@keyframes reveal-a { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes reveal-b { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.result-card.run-a { animation: reveal-a .45s ease-out both; }
.result-card.run-b { animation: reveal-b .45s ease-out both; }
.result-card {
  display: flex; align-items: center; gap: 1.3rem;
  background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.result-card .label { color: var(--muted); font-size: .95rem; }
.result-card .name { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.9rem; font-weight: 800; line-height: 1.15; }

/* Confidence ring: counts up from 0 */
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
@keyframes fill { from { --p: 0; } to { --p: var(--target); } }
.ring {
  --target: 0; --p: 0;
  position: relative; display: grid; place-items: center;
  width: 96px; height: 96px; flex: none; border-radius: 50%;
  background: conic-gradient(var(--ring-color, var(--accent)) calc(var(--p) * 1%), var(--soft) 0);
  animation: fill 1s ease-out .1s both;
}
.ring::before { content: ""; position: absolute; width: 74px; height: 74px; border-radius: 50%; background: var(--card); }
.ring span { position: relative; font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.2rem; }
.ring.low { --ring-color: var(--warn); }

/* Top-5 bars */
@keyframes grow { from { width: 0; } to { width: var(--w); } }
.bar-row { margin: 0 0 .9rem 0; }
.bar-head { display: flex; justify-content: space-between; margin-bottom: .3rem; }
.bar-row.top .bar-head { font-weight: 600; }
.bar-track { height: 8px; background: var(--soft); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: #6f9a80; width: var(--w); animation: grow .8s ease-out both; animation-delay: var(--d); }
.bar-row.top .bar-fill { background: var(--accent); }

/* Empty state */
.empty { text-align: center; color: var(--muted); padding: 3rem 1rem; border: 1px dashed #5f7a67; border-radius: 12px; background: var(--card); }
.empty .big { font-size: 2rem; margin-bottom: .3rem; }

/* Benchmark dots */
.dots { display: flex; gap: 6px; margin: .4rem 0 1.2rem; flex-wrap: wrap; }
.dot { width: 12px; height: 12px; border-radius: 50%; background: var(--soft); }
.dot.on { background: var(--accent); }

@media (max-width: 640px) {
  .page-title { font-size: 1.9rem; }
  .result-card { flex-direction: column; align-items: flex-start; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .ring { --p: var(--target); }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# Loaders
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_species_info():
    # Not cached, so a missing file is reported and fixed without clearing cache.
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
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + title.replace(" ", "_")
        )
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "NatureClassificationApp/1.0"},
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


def pretty(name):
    return name.replace("_", " ").title()


def md_html(markup):
    # Markdown treats 4+ leading spaces as a code block, so strip indentation.
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------
# Components
# -----------------------------
def render_result_card(result):
    confidence = result["confidence"]
    low = confidence < LOW_CONFIDENCE
    run_class = "run-a" if result["run_id"] % 2 == 0 else "run-b"
    name = html.escape(pretty(result["class"]))

    md_html(
        f"""
        <div class="result-card {run_class}">
          <div class="ring {'low' if low else ''}" style="--target:{confidence:.1f}">
            <span>{confidence:.0f}%</span>
          </div>
          <div>
            <div class="label">Predicted class</div>
            <div class="name">{name}</div>
          </div>
        </div>
        """
    )

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Confidence", f"{confidence:.2f}%")
    metric_col2.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")

    if low:
        st.warning(
            "The model is not very sure about this image. "
            "Try a clearer photo of a single animal or plant."
        )


def render_top_predictions(result, class_names):
    scores = np.array(result["predictions"])
    rows = []
    for rank, index in enumerate(np.argsort(scores)[-5:][::-1]):
        score = float(scores[index]) * 100
        rows.append(
            f"""
            <div class="bar-row {'top' if rank == 0 else ''}">
              <div class="bar-head"><span>{html.escape(pretty(class_names[index]))}</span>
              <span>{score:.2f}%</span></div>
              <div class="bar-track">
                <div class="bar-fill" style="--w:{min(score, 100):.2f}%; --d:{rank * 0.12:.2f}s"></div>
              </div>
            </div>
            """
        )
    st.markdown("##### Top 5 Predictions")
    md_html("".join(rows))


def show_education_page(class_key, species_info):
    pretty_name = pretty(class_key)
    info = species_info.get(class_key)

    if info:
        st.header(f"📚 {info['common_name']}")
        st.caption(f"*{info['scientific_name']}*")

        col1, col2 = st.columns(2)
        col1.metric("Type", info["type"])
        col2.metric("Wild / Cultivated", info["wild_status"])

        status_text = f"Conservation status: {info['conservation_status']}"
        level = info.get("status_level", "unknown")
        if level == "threatened":
            st.error("⚠️ " + status_text)
        elif level == "safe":
            st.success("✅ " + status_text)
        else:
            st.info("ℹ️ " + status_text)

        st.subheader("About")
        st.write(info["brief"])

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
            st.warning(
                "No information available yet. Add it to data/species_info.json."
            )


def empty_state(icon, text):
    st.markdown(
        f'<div class="empty"><div class="big">{icon}</div>{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Header + model loading
# -----------------------------
md_html(
    """
    <h1 class="page-title">🌿 Nature Classification</h1>
    <p class="page-sub">Upload a photo of an animal or plant to identify it, then read a short guide about the species.</p>
    """
)

try:
    with st.spinner("Loading model..."):
        model = load_model()
        class_names = load_class_names()
        species_info = load_species_info()
    st.caption("✅ AI model loaded successfully!")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

if "timings_ms" not in st.session_state:
    st.session_state["timings_ms"] = []
if "run_count" not in st.session_state:
    st.session_state["run_count"] = 0

tab_identify, tab_bench = st.tabs(["Identify", "Benchmark"])

# -----------------------------
# Identify tab
# -----------------------------
with tab_identify:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Your photo")
            uploaded_file = st.file_uploader(
                "Choose an image (JPG or PNG)",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )

            image = None
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception as error:
                    st.error(f"Could not open this image: {error}")
                    st.stop()

                st.image(image, use_container_width=True)
                classify = st.button(
                    "Classify image", type="primary", use_container_width=True
                )
            else:
                classify = False

    if classify and image is not None:
        with st.spinner("Analyzing image..."):
            # Measures preprocessing + model prediction only.
            start_time = time.perf_counter()

            resized_image = image.resize(IMAGE_SIZE)
            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)
            predictions = model.predict(image_array, verbose=0)

            processing_time_ms = (time.perf_counter() - start_time) * 1000

        predicted_index = int(np.argmax(predictions[0]))
        st.session_state["run_count"] += 1
        st.session_state["result"] = {
            "predictions": predictions[0].tolist(),
            "index": predicted_index,
            "confidence": float(predictions[0][predicted_index]) * 100,
            "class": class_names[predicted_index],
            "processing_time_ms": processing_time_ms,
            "run_id": st.session_state["run_count"],
        }

        timings = st.session_state["timings_ms"]
        timings.append(processing_time_ms)
        st.session_state["timings_ms"] = timings[-TIMING_SAMPLE_LIMIT:]

    result = st.session_state.get("result")

    with right:
        if uploaded_file is not None and result:
            st.markdown("#### Result")
            render_result_card(result)
            render_top_predictions(result, class_names)
        else:
            empty_state("🔍", "Upload a photo and select Classify image to see the result here.")

    if uploaded_file is not None and result:
        st.divider()
        show_education_page(result["class"], species_info)


# -----------------------------
# Benchmark tab
# -----------------------------
with tab_bench:
    timings = st.session_state["timings_ms"]

    if not timings:
        empty_state("⏱️", "No runs yet. Classify images to build a 20-run sample.")
    else:
        st.caption(
            f"Recorded {len(timings)} of {TIMING_SAMPLE_LIMIT} runs. "
            "Classify different images to build the sample."
        )

        dots = "".join(
            f'<div class="dot {"on" if i < len(timings) else ""}"></div>'
            for i in range(TIMING_SAMPLE_LIMIT)
        )
        st.markdown(f'<div class="dots">{dots}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Average", f"{float(np.mean(timings)):.2f} ms")
        c2.metric("Fastest", f"{float(np.min(timings)):.2f} ms")
        c3.metric("Slowest", f"{float(np.max(timings)):.2f} ms")

        peak = max(timings) or 1.0
        bars = "".join(
            f'<div class="run" title="Run {i}: {t:.2f} ms">'
            f'<div class="val">{t:.0f}</div>'
            f'<div class="col" style="--h:{max(t / peak * 100, 3):.1f}%; --d:{i * 0.03:.2f}s"></div>'
            f'<div class="n">{i}</div></div>'
            for i, t in enumerate(timings, start=1)
        )
        st.markdown("##### Time per run (ms)")
        md_html(f'<div class="runs">{bars}</div>')

        if len(timings) == TIMING_SAMPLE_LIMIT:
            st.success(f"Benchmark sample complete: {TIMING_SAMPLE_LIMIT} classification runs.")
        else:
            st.info(
                f"{TIMING_SAMPLE_LIMIT - len(timings)} more run(s) needed "
                "to complete the 20-run sample."
            )

        with st.expander("View individual run times"):
            for run_number, run_time in enumerate(timings, start=1):
                st.write(f"Run {run_number}: {run_time:.2f} ms")

        if st.button("Reset benchmark"):
            st.session_state["timings_ms"] = []
            st.rerun()