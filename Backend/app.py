import os
from pathlib import Path
from flask import Flask, request, jsonify  # Flask web framework
from flask_cors import CORS  # Cross-Origin Resource Sharing
import numpy as np  # Numerical operations
import cv2  # OpenCV for image processing

# List of Arabic characters
arabic_characters = ["أ Alif", "ب Bae", "ت Tae", "ث Thae", "ج Jim ", "ح Hae ", "خ Khae", " Dal د", "Zal ذ", "Rae ر", "Zay ز", " Sin س", "Shin ش", " Sad ص", "Dad ض", "ط Tae ", "ظ Dad ", "Ayn ع", "Ghayn غ", "Fae ف", "Kaf ق", " ك Kaf ", "ل Lam ", "م Mim", "نNoun ", "Hae ه", "و Waw ", "Yae ي"]

def create_app(model=None):
    """Create the API with a checkpoint, or an injected predictor for tests."""
    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    if model is None:
        model_path = Path(os.environ.get("ARABIC_CNN_MODEL_PATH",
            str(Path(__file__).resolve().parents[1] / "Models" / "model4arabic.keras")))
        if not model_path.is_file():
            raise FileNotFoundError("Export a trained Keras model and set ARABIC_CNN_MODEL_PATH to its location.")
        from tensorflow.keras.models import load_model
        model = load_model(model_path)

    @app.route('/convert', methods=['POST'])
    def convert_image():
        file = request.files.get('file')
        if file is None:
            return jsonify({'error': 'No file provided'}), 400
        try:
            image_bytes = file.read()
            if not image_bytes:
                return jsonify({'error': 'Empty image'}), 400
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return jsonify({'error': 'Invalid image'}), 400
            resized = cv2.resize(image, (32, 32)).astype('float32') / 255.0
            prediction = np.asarray(model.predict(resized[None, ..., None], verbose=0))
            if prediction.shape != (1, 28) or not np.isfinite(prediction).all():
                raise ValueError("Model output must contain 28 finite class scores")
            return jsonify({'message': arabic_characters[int(prediction.argmax(axis=-1)[0])]})
        except Exception:
            app.logger.exception('Image inference failed')
            return jsonify({'error': 'Image inference failed'}), 500
    return app


if __name__ == '__main__':
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1")
