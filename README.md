# AraboScripto — Arabic Handwritten Character Recognition

A TensorFlow/Keras CNN prototype for classifying images of **isolated Arabic handwritten characters into 28 classes**. The repository includes a training notebook, a Flask inference endpoint and React interface source. It is not a full-word or sentence transcription system.

## Project question and deliverable

How can a handwritten character image be converted into one of 28 Arabic character labels? This project connects a CNN training notebook to an image-upload interface and an inference endpoint.

The key integration requirement is consistency: image orientation, grayscale conversion, 32 × 32 resizing, normalization, and class-index mapping must agree between the notebook and backend. A model can produce a confident prediction even when one of these conventions is wrong.

**Start here:** inspect the [training notebook](Models/) for the classifier, then [Backend/app.py](Backend/app.py) for the preprocessing and response contract. The notebook is available to review; running the complete application also requires the missing artifacts listed below.

## Implementation

The notebook builds a convolutional classifier with pooling, batch normalization, dropout and a softmax output. The Flask backend decodes an uploaded image as grayscale, resizes it to **32 × 32**, scales pixel values to 0–1 and maps the predicted class to a character label.

## What is present and missing

| Item | Status |
|---|---|
| Training notebook | Present in `Models/` |
| Character dataset files | Present in `my_dataset/` |
| Flask source | Present in `Backend/app.py` |
| React source | Present under `Frontend/my-react-app- main/react-app-main/` |
| Exported `model4arabic.keras` | Not included |
| Python dependency manifest | Not included |
| React `package.json` and lockfile | Not included |

The app cannot be reproduced end to end until the original model export and environment/manifests are restored. The directory names and capitalization above match the checkout.

## Backend configuration

The backend depends on Flask, Flask-CORS, TensorFlow/Keras, NumPy and OpenCV. Compatible versions are not pinned here; restore the original environment before treating it as reproducible.

With those dependencies and a compatible model available, run from the repository root:

```bash
export ARABIC_CNN_MODEL_PATH="/absolute/path/to/model4arabic.keras"
python Backend/app.py
```

The default model location is `Models/model4arabic.keras`. A missing file produces an actionable startup error. Debug mode is off by default; `FLASK_DEBUG=1` enables it for local development.

The inference route is `POST /convert`, with a multipart form field named `file`. A successful response includes a `message` containing the character label. For example, against a running local server:

```bash
curl -F "file=@/absolute/path/to/character.png" http://127.0.0.1:5000/convert
```

This is a request example, not a verified prediction result. The React dependency manifests are missing, so a frontend launch command is intentionally not supplied as a working setup.

## Training and metric interpretation

The notebook contains developer-specific Windows paths. Replace these with your local dataset locations, verify image orientation and class indices, then export a compatible Keras model.

The notebook also adds a misclassified test image to training and reevaluates on that test set. Results from that revised run are contaminated. Neither the previous README's 94.7% figure nor the portfolio's 92.4% figure is certified as independent test accuracy here. Preserve the original notebook as a historical record and use a fresh untouched holdout for a new claim.

## Limitations and next steps

Inputs are isolated characters, not connected handwriting. Resizing, orientation and label indexing must match training. Uploaded-image inference is not equivalent to a standardized dataset benchmark. Restore manifests and checkpoint, make notebook paths portable, document dataset provenance, split training/validation/test data before iteration, and report confusion matrices and per-class errors on the final holdout.
