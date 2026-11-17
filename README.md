# 🌿 Nature Classification

A web app that identifies the category of an animal or plant from an uploaded photo. It uses a trained deep learning model (TensorFlow/Keras) and a simple [Streamlit](https://streamlit.io/) interface.

Upload an image, click **Classify Image**, and the app shows the predicted class, a confidence score, and the top 5 most likely classes. The app also provides educational information about the predicted species when available locally or through Wikipedia.

---

## ✨ Features

- Upload images in `.jpg`, `.jpeg` or `.png` format
- Image preview before classification
- Predicted class with confidence percentage
- Top 5 predictions, each with a progress bar
- Species education page with habitat, threats, conservation status, and fun facts when available
- Wikipedia fallback for classes without local species information
- Model and class names are cached for faster responses
- Clear error message if the model fails to load

---

## 🗂️ Project Structure

```text
Nature_Classification/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore
├── model/
│   ├── nature_classification_final.keras   # Trained model
│   └── class_names.json                    # List of class labels
└── data/
    └── species_info.json                   # Educational information by species
```

---

## 🛠️ Tech Stack

| **Purpose** | **Library** |
|---|---|
| Deep learning | TensorFlow / Keras |
| Web interface | Streamlit |
| Image handling | Pillow |
| Array operations | NumPy |
| External species information | Wikipedia REST API |

---

## 📊 Dataset and Model Performance

The model is based on **MobileNetV2** and classifies images into **27 plant and animal categories**.

| **Metric** | **Result** |
|---|---:|
| Dataset images | 8,954 |
| Training images | 6,302 |
| Validation images | 1,352 |
| Test images | 1,300 |
| Test accuracy | 79.46% |
| Macro F1-score | 77.74% |
| Weighted F1-score | 79.56% |
| Model size | 21.11 MiB |
| Average single-image inference latency | 81.47 ms |
| Batch throughput (batch size 32) | 228.56 images/second |

**Evaluation note:** Accuracy and F1-scores were calculated on the project's held-out test split. Inference latency and throughput are separate benchmarks measured in the Colab environment. The throughput benchmark used repeated copies of an image and should be interpreted as a batch benchmark, not end-to-end Streamlit response time.

### Dataset quality checks

- Checked all 8,954 images using SHA1 hashes: no exact duplicate files were detected.
- Checked all 8,954 images using perceptual hashing (pHash, distance threshold ≤ 5): no similar image pairs were detected.
- No exact or perceptually similar cross-split pairs were detected by these checks.

These checks reduce the risk of duplicate-image leakage but do not guarantee the absence of every visually or semantically related image.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Akrist3/Nature_Classification.git
cd Nature_Classification
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

---

## 🧠 How It Works

1. The app loads the trained model from `model/nature_classification_final.keras` and the class labels from `model/class_names.json`.
2. The uploaded image is converted to RGB and resized to **224 × 224** pixels.
3. The image is passed to the MobileNetV2-based model, which outputs a probability for each of the 27 classes.
4. The class with the highest probability is shown as the prediction, and the five highest-scoring classes are listed with their confidence.
5. The app displays educational information from `data/species_info.json`, or uses Wikipedia as a fallback when local information is unavailable.

> **Note:** The model file and `class_names.json` must be present in the `model/` folder for the app to start. The `data/species_info.json` file is optional; the app can fall back to Wikipedia summaries.

---

## 📸 Usage

1. Open the app.
2. Click **Browse files** and choose an image of an animal or plant.
3. Click **Classify Image**.
4. View the predicted class, confidence score and top 5 predictions.
5. Read the educational information shown below the prediction.

For best results, use a clear, well-lit photo where the subject fills most of the frame.

---

## ⚠️ Limitations

- The model can only predict the classes it was trained on. Images of anything else will still be assigned the closest known class.
- Accuracy depends on image quality, lighting and angle.
- Confidence scores show the model's output probability, not a guarantee that the prediction is correct.
- Educational information may be unavailable for some classes, and Wikipedia fallback content depends on network access.
- The reported inference benchmark does not represent total browser-to-browser application response time.

---

## 🔮 Future Improvements

- Improve classification performance, especially for classes with lower F1-scores
- Support camera input for live capture
- Deploy to Streamlit Community Cloud or Hugging Face Spaces
- Expand the dataset and add more plant and animal categories
- Measure end-to-end application response time under real usage conditions

---

## 👤 Author

**Akrist3** — [GitHub Profile](https://github.com/Akrist3)

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (for example MIT) to define how others may use this project.
