import unittest
import gc
import torch
from PIL import Image
from pipeline.orchestrator import LayoutPredictor, LOCAL_LAYOUT_MODEL

class TestLayoutPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = LayoutPredictor(LOCAL_LAYOUT_MODEL)

    @classmethod
    def tearDownClass(cls):
        cls.predictor.model = None
        cls.predictor.image_processor = None
        cls.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_predictor_initialization_and_prediction(self):
        # 验证模型和图像处理器被成功加载且不为空
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.image_processor)
        
        # 运行测试预测
        test_img = Image.new("RGB", (300, 300), (255, 255, 255))
        results = self.predictor.predict(test_img)
        self.assertEqual(len(results), 1)
        
        # 严格断言：对结果字典的结构与数据类型进行校验
        result = results[0]
        self.assertIn("scores", result)
        self.assertIn("labels", result)
        self.assertIn("boxes", result)
        
        self.assertTrue(isinstance(result["scores"], torch.Tensor))
        self.assertTrue(isinstance(result["labels"], torch.Tensor))
        self.assertTrue(isinstance(result["boxes"], torch.Tensor))
