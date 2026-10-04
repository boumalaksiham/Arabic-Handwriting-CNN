import io
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from Backend.app import create_app, arabic_characters

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        scores = np.zeros((1, 28)); scores[0, 2] = 1
        self.model.predict.return_value = scores
        self.client = create_app(self.model).test_client()

    def test_missing_empty_and_invalid_images_are_client_errors(self):
        self.assertEqual(self.client.post('/convert').status_code, 400)
        for payload in (b'', b'not an image'):
            response = self.client.post('/convert', data={'file': (io.BytesIO(payload), 'input.png')})
            self.assertEqual(response.status_code, 400)
        self.model.predict.assert_not_called()

    def test_image_shape_normalization_and_class_mapping(self):
        _, encoded = cv2.imencode('.png', np.full((40, 50), 255, np.uint8))
        result = self.client.post('/convert', data={'file': (io.BytesIO(encoded.tobytes()), 'input.png')})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json['message'], arabic_characters[2])
        tensor = self.model.predict.call_args.args[0]
        self.assertEqual(tensor.shape, (1, 32, 32, 1))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor == 1))

    def test_bad_checkpoint_output_returns_generic_error(self):
        self.model.predict.return_value = np.zeros((1, 29))
        _, encoded = cv2.imencode('.png', np.zeros((32, 32), np.uint8))
        result = self.client.post('/convert', data={'file': (io.BytesIO(encoded.tobytes()), 'input.png')})
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.json, {'error': 'Image inference failed'})

if __name__ == '__main__':
    unittest.main()
