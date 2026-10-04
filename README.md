# AraboScripto — Arabic Handwritten Character Recognition

A TensorFlow/Keras CNN prototype for classifying images of **isolated Arabic handwritten characters into 28 classes**. The repository includes a training notebook, a Flask inference endpoint and React interface source. It is not a full-word or sentence transcription system.

## Project question and deliverable

How can a handwritten character image be converted into one of 28 Arabic character labels? This project connects a CNN training notebook to an image-upload interface and an inference endpoint.

The key integration requirement is consistency: image orientation, grayscale conversion, 32 × 32 resizing, normalization, and class-index mapping must agree between the notebook and backend. A model can produce a confident prediction even when one of these conventions is wrong.

**Start here:** inspect the portable [training script](train.py) for the classifier and the [historical notebook](Models/) for the earlier experiments, then [Backend/app.py](Backend/app.py) for the preprocessing and response contract. The notebook is available to review; running the complete application also requires the missing artifacts listed below.

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
| Python dependency manifest | Present in `requirements.txt`; supported ranges, not an exact historical environment |
| React `package.json` and lockfile | Restored; production build verified |

A real inference session still requires a trained checkpoint. The portable trainer can produce a new export; it does not reproduce or certify the notebook’s historical scores. The directory names and capitalization above match the checkout.

## Backend configuration

The backend depends on Flask, Flask-CORS, TensorFlow/Keras, NumPy and OpenCV. Version ranges are provided in `requirements.txt`. Use a Python 3.12 virtual environment; TensorFlow training has not been rerun during this repair.

With those dependencies and a compatible model available, run from the repository root:

```bash
python -m pip install -r requirements.txt
export ARABIC_CNN_MODEL_PATH="/absolute/path/to/model4arabic.keras"
python Backend/app.py
```

The default model location is `Models/model4arabic.keras`. A missing file produces an actionable startup error. Debug mode is off by default; `FLASK_DEBUG=1` enables it for local development.

The inference route is `POST /convert`, with a multipart form field named `file`. A successful response includes a `message` containing the character label. For example, against a running local server:

```bash
curl -F "file=@/absolute/path/to/character.png" http://127.0.0.1:5000/convert
```

Invalid or empty images return HTTP 400. Uploads are capped at 8 MiB. Model outputs must contain 28 finite scores; inference errors return a generic HTTP 500 response.

## Frontend setup

```bash
cd "Frontend/my-react-app- main/react-app-main"
npm ci
npm start
# For a production build:
npm run build
```

The restored manifest matches the existing Create React App source. A production build passed during this repair. The interface calls `http://localhost:5000/convert` by default; set `REACT_APP_API_URL` to another backend base URL before starting or building. This build check does not exercise a real checkpoint or uploaded-character predictions.

## Training and metric interpretation

The notebook retains its original Windows paths as a historical record. The new `train.py` uses repository-relative paths, validates CSV dimensions/pixels/labels, and selects early stopping using a stratified 20% split from training rows. The supplied test rows are used only after fitting and checkpoint export.

```bash
python train.py --epochs 30 --seed 42
export ARABIC_CNN_MODEL_PATH="$PWD/Models/retrained/model4arabic.keras"
python Backend/app.py
```

The trainer exports a Keras checkpoint and `evaluation.json` containing split indices, dataset hashes, training history, test predictions, a confusion matrix, and per-class metrics. It retains the notebook’s CNN layer layout but uses a documented fresh training configuration without the notebook’s test-image retraining step. No new accuracy result is claimed until this script is run.

CSV loading was checked on all 13,440 training and 3,360 test rows; labels mapped to indices 0–27. Verify orientation against sample characters before using a new checkpoint.

The notebook also adds a misclassified test image to training and reevaluates on that test set. Results from that revised run are contaminated. Neither the previous README's 94.7% figure nor the portfolio's 92.4% figure is certified as independent test accuracy here. Preserve the original notebook as a historical record and use a fresh untouched holdout for a new claim.

## Limitations and next steps

Inputs are isolated characters, not connected handwriting. Resizing, orientation and label indexing must match training. Uploaded-image inference is not equivalent to a standardized dataset benchmark. Train and inspect a checkpoint, document dataset provenance, and obtain fresh held-out data for any independent generalization claim. The supplied test split was already examined in historical work.

## API regression checks

```bash
python -m pip install Flask flask-cors numpy opencv-python
python -m unittest discover -s tests -v
```

Three checks cover missing/invalid uploads, grayscale tensor shape and normalization, class-index mapping, and malformed model outputs. They inject a controlled predictor; they do not measure CNN accuracy or need TensorFlow/a checkpoint.
