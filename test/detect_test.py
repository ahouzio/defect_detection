import unittest
import torch
from src.detect import detection_model

class TestDetectionModel(unittest.TestCase):
    def setUp(self):
        self.model_path = "artifacts/defect_5-v0/best.pt"
        self.size = 256 # size used to train my model
        self.img_path = "test_images/20.jpg"
        self.conf = 0.25
        self.iou = 0.45
        self.max_det = 1
        self.amp = False
        self.model = detection_model(self.model_path, self.conf, self.iou, self.max_det, self.amp)
        self.img_dir = "test_images"
        
    def test_init(self):
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model.model, torch.nn.Module)
        self.assertEqual(self.conf, self.conf)
        self.assertEqual(self.iou, self.iou)
        self.assertEqual(self.max_det, self.max_det)
        self.assertEqual(self.amp, self.amp)

    def test_predict(self):
        prediction = self.model.predict(self.img_path, self.size)
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), 6)
        self.assertIsInstance(prediction[0], float)
        self.assertIsInstance(prediction[1], float)
        self.assertIsInstance(prediction[2], float)
        self.assertIsInstance(prediction[3], float)
        self.assertEqual(prediction[4], 20)
        self.assertIsInstance(prediction[5], float)
        
    def test_evaluate(self):
        class_ratio, weighted_acc = self.model.evaluate(self.img_dir, self.size)
        self.assertIsInstance(class_ratio, float)
        self.assertIsInstance(weighted_acc, float)
        self.assertGreaterEqual(class_ratio, 0)
        self.assertLessEqual(class_ratio, 1)
        self.assertGreaterEqual(weighted_acc, 0)
        self.assertLessEqual(weighted_acc, 1)