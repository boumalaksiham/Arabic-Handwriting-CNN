#  AraboScripto — Arabic Handwritten Character Recognition

## Published-source status

The implementation uses **TensorFlow/Keras**, not PyTorch. The opening accuracy figure below is reported training accuracy, not verified held-out test accuracy. The backend expects a trained `.keras` model that is not included in this checkout; set `ARABIC_CNN_MODEL_PATH` to your exported model. The notebook also adds a misclassified test image to training data and reevaluates on that test set. Metrics from that revised run are contaminated and should not be presented as untouched-test accuracy. Preserve the notebook as a historical record and rerun with a fresh holdout before publishing a test claim. The React source is present, but its `package.json` and lockfile are missing, so the frontend cannot yet be reproduced with `npm install`. Restore the original dependency manifests before launching it.


> **Deep learning web application** — a CNN-powered system that recognizes handwritten Arabic characters from uploaded images, reporting 94.7% training accuracy across 28 character classes.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-FF6F00?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.0-D00000?style=flat-square&logo=keras)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)

---

##  What is AraboScripto?

Arabic handwriting recognition is a significantly harder problem than Latin script recognition due to the cursive nature of Arabic, its right-to-left directionality, and the high visual similarity between certain characters.

AraboScripto is a full-stack web application that lets users upload a PNG image of a handwritten Arabic character and instantly receive the recognized character name — powered by a custom-trained Convolutional Neural Network.

---

##  System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    React.js Frontend                         │
│    Upload interface │ Image preview │ Recognition result     │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP POST (image)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Flask Backend (app.py)                     │
│         Image validation, preprocessing, model inference     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    CNN Model (Keras)                         │
│                                                             │
│  Input (image)                                              │
│       │                                                     │
│       ▼                                                     │
│  Conv2D → MaxPooling2D   (feature extraction layer 1)       │
│       │                                                     │
│       ▼                                                     │
│  Conv2D → MaxPooling2D   (feature extraction layer 2)       │
│       │                                                     │
│       ▼                                                     │
│  Conv2D → MaxPooling2D   (feature extraction layer 3)       │
│       │                                                     │
│       ▼                                                     │
│  Flatten → Dense → Softmax                                  │
│       │                                                     │
│       ▼                                                     │
│  Predicted character (28 classes)                           │
└─────────────────────────────────────────────────────────────┘
```

---

##  CNN Model Architecture

The model was designed, trained, and evaluated in Jupyter Notebook using Keras and TensorFlow on the AHCD (Arabic Handwritten Characters Dataset) from Kaggle.
```
Input Layer       → grayscale image (32×32×1)
Conv2D (32)       → ReLU activation
MaxPooling2D      → spatial downsampling
Conv2D (64)       → ReLU activation
MaxPooling2D      → spatial downsampling
Conv2D (128)      → ReLU activation
MaxPooling2D      → spatial downsampling
Flatten
Dense (128)       → ReLU activation
Dense (28)        → Softmax (one class per Arabic character)
```

### Training Results

| Metric | Value |
|--------|-------|
| Dataset | AHCD — Arabic Handwritten Characters |
| Classes | 28 Arabic characters |
| Training accuracy | 94.7% |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |

---

##  Features

- **Image upload** — drag and drop or click to upload PNG images of handwritten Arabic characters
- **Live preview** — users see their uploaded image before submitting
- **Instant recognition** — CNN inference returns the character name in real time
- **Image swap** — users can replace their image and re-run recognition without refreshing
- **Clean UI** — informational landing page + dedicated recognition interface

---

##  Key Engineering Decisions

**Why a custom CNN over a pre-trained model?**
Building from scratch on Arabic handwriting data allowed full control over the architecture and gave a deeper understanding of how convolutional layers learn spatial features specific to Arabic character shapes. Transfer learning is a planned next step.

**Why three convolutional layers?**
Single-layer CNNs struggle with the fine-grained stroke differences between visually similar Arabic characters (e.g., ب، ت، ث). Three stacked Conv+Pool blocks allow the network to learn increasingly abstract representations — from edges in layer 1 to character-level patterns in layer 3.

**Why Flask + React?**
Flask handles model inference efficiently as a lightweight API. React provides a responsive, component-based frontend with clean state management for the upload → preview → result flow.

---

##  Project Structure
```
araboscripto/
├── backend/
│   ├── app.py               # Flask API — image upload, preprocessing, inference
│   ├── model/
│   │   └── arabic_cnn.h5    # Trained Keras model
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── UploadPage.js    # Image upload + preview + result display
│   │   └── LandingPage.js   # About page
│   └── package.json
└── notebook/
    └── arabic_cnn_training.ipynb   # Full training pipeline
```

---

##  Quick Start

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

Open **http://localhost:3000** in your browser.

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js, HTML, CSS |
| Backend | Python, Flask |
| Deep Learning | TensorFlow, Keras |
| Data Analysis | pandas, NumPy, matplotlib |
| Model Training | Jupyter Notebook, Anaconda |
| Dataset | AHCD — Kaggle |

---

##  Roadmap

- [ ] Transfer learning using pre-trained CNN (VGG16, ResNet) for improved accuracy
- [ ] Data augmentation (rotation, shear, zoom) to reduce overfitting
- [ ] Expand to full word and sentence recognition
- [ ] Support for additional image formats (JPG, TIFF)
- [ ] Mobile-responsive UI
- [ ] Model confidence score displayed alongside prediction

---

##  Author

**Siham Boumalak**
B.A. Computer Science & Data Science — The College of Wooster
M.S. Artificial Intelligence — Northeastern University, Khoury College | Expected 2027

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/siham-boumalak/)
[![GitHub](https://img.shields.io/badge/GitHub-boumalaksiham-181717?style=flat-square&logo=github)](https://github.com/boumalaksiham)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20AHCD-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/mloey1/ahcd1)

## Backend environment configuration

Export a trained model from the notebook, then set an absolute path before starting the backend:

```bash
export ARABIC_CNN_MODEL_PATH="/absolute/path/to/model4arabic.keras"
python Backend/app.py
```

The model is not distributed in this checkout. Dependencies and the frontend manifests still need restoration; this command documents the model path rather than promising a complete setup. Debug mode is off by default; set `FLASK_DEBUG=1` only for local development.
