# 🌿 Nature Classification

A web app that identifies the category of an animal or plant from an uploaded photo. It uses a trained deep learning model (TensorFlow/Keras) and a simple [Streamlit](https://streamlit.io/) interface.

Upload an image, click **Classify Image**, and the app shows the predicted class, a confidence score, and the top 5 most likely classes.

---

## ✨ Features

- Upload images in `.jpg`, `.jpeg` or `.png` format
- Image preview before classification
- Predicted class with confidence percentage
- Top 5 predictions, each with a progress bar
- Model and class names are cached for faster responses
- Clear error message if the model fails to load

---

## 🗂️ Project Structure

```
Nature_Classification/
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore
└── model/
    ├── nature_classification_final.keras   # Trained model
    └── class_names.json                    # List of class labels
```

---

## 🛠️ Tech Stack

| Purpose | Library |
|---|---|
| Deep learning | TensorFlow / Keras |
| Web interface | Streamlit |
| Image handling | Pillow |
| Array operations | NumPy |

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
3. The image is passed to the model, which outputs a probability for each class.
4. The class with the highest probability is shown as the prediction, and the five highest-scoring classes are listed with their confidence.

> **Note:** The model file and `class_names.json` must be present in the `model/` folder for the app to start.

---

## 📸 Usage

1. Open the app.
2. Click **Browse files** and choose an image of an animal or plant.
3. Click **Classify Image**.
4. View the predicted class, confidence score and top 5 predictions.

For best results, use a clear, well-lit photo where the subject fills most of the frame.

---

## ⚠️ Limitations

- The model can only predict the classes it was trained on. Images of anything else will still be assigned the closest known class.
- Accuracy depends on image quality, lighting and angle.
- Confidence scores show how sure the model is, not a guarantee that the prediction is correct.

---

## 🔮 Future Improvements

- Add model accuracy metrics and a confusion matrix
- Support camera input for live capture
- Deploy to Streamlit Community Cloud or Hugging Face Spaces
- Add more classes and training data

---

## 👤 Author

**Akrist3** — [GitHub Profile](https://github.com/Akrist3)

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (for example MIT) to define how others may use this project.
