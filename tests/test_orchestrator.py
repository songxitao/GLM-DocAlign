import unittest
import gc
import torch
from unittest.mock import patch
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


import tempfile
import shutil
import asyncio
from pathlib import Path
from pipeline.orchestrator import StreamAssembler

class TestStreamAssembler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_assembler_order_preservation(self):
        async def run_test():
            assembler = StreamAssembler(self.temp_dir, "test_doc", total_pages=2)
            
            # 页 0 骨架（1个 OCR，1个直接截图块）
            page_0 = {
                "page_idx": 0,
                "page_size": [100, 100],
                "blocks": [
                    {"block_idx": 0, "type": "ocr_task", "label": "text", "bbox": [0,0,10,10], "content": None},
                    {"block_idx": 1, "type": "image_block", "label": "figure", "bbox": [10,10,20,20], "image_path": "images/test_0.png"}
                ]
            }
            # 页 1 骨架（1个 OCR）
            page_1 = {
                "page_idx": 1,
                "page_size": [100, 100],
                "blocks": [
                    {"block_idx": 0, "type": "ocr_task", "label": "text", "bbox": [0,0,10,10], "content": None}
                ]
            }
            
            # 1. 注册页 1
            await assembler.register_page(1, page_1)
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 2. 注册页 0
            await assembler.register_page(0, page_0)
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 3. 填充页 1 OCR（乱序返回）
            await assembler.fill_ocr_content(1, 0, "Hello Page 1")
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 4. 填充页 0 OCR
            await assembler.fill_ocr_content(0, 0, "Hello Page 0")
            await asyncio.wait_for(assembler.finished_event.wait(), timeout=2.0)
            self.assertEqual(assembler.current_writing_page, 2)
            
            # 验证写入的物理文件内容是否符合预期
            output_md = Path(self.temp_dir) / "final_output.md"
            self.assertTrue(output_md.exists())
            content = output_md.read_text(encoding="utf-8")
            self.assertIn("Hello Page 0", content)
            self.assertIn("Hello Page 1", content)
            
        asyncio.run(run_test())

    def test_assembler_idempotency_and_memory_clean(self):
        async def run_test():
            assembler = StreamAssembler(self.temp_dir, "test_doc", total_pages=1)
            
            # 页 0 骨架（2个 OCR 块）
            page_0 = {
                "page_idx": 0,
                "page_size": [100, 100],
                "blocks": [
                    {"block_idx": 0, "type": "ocr_task", "label": "text", "bbox": [0,0,10,10], "content": None},
                    {"block_idx": 1, "type": "ocr_task", "label": "text", "bbox": [10,10,20,20], "content": None}
                ]
            }
            
            # 注册页 0
            await assembler.register_page(0, page_0)
            self.assertEqual(assembler.page_buffers[0]["pending_ocr"], 2)
            self.assertFalse(assembler.page_buffers[0]["ready"])
            
            # 1. 填充不存在的 block_idx (例如 99)，不发生状态扣减
            await assembler.fill_ocr_content(0, 99, "No Match")
            self.assertEqual(assembler.page_buffers[0]["pending_ocr"], 2)
            self.assertFalse(assembler.page_buffers[0]["ready"])
            
            # 2. 填充 block 0，pending_ocr 减一
            await assembler.fill_ocr_content(0, 0, "Hello 0")
            self.assertEqual(assembler.page_buffers[0]["pending_ocr"], 1)
            self.assertFalse(assembler.page_buffers[0]["ready"])
            
            # 3. 重复填充 block 0 (幂等保护测试)，pending_ocr 应保持为 1，且不应触发报错
            await assembler.fill_ocr_content(0, 0, "Hello 0 Duplicate")
            self.assertEqual(assembler.page_buffers[0]["pending_ocr"], 1)
            self.assertFalse(assembler.page_buffers[0]["ready"])
            
            # 4. 填充 block 1，页面 ready，触发写盘
            await assembler.fill_ocr_content(0, 1, "Hello 1")
            await asyncio.wait_for(assembler.finished_event.wait(), timeout=2.0)
            
            # 验证内存清理：已写盘页面的 page_buffers 应该被删除
            self.assertNotIn(0, assembler.page_buffers)
            
            # 验证 final_pdf_info 正确保存了页面结构
            self.assertEqual(len(assembler.final_pdf_info), 1)
            self.assertIsNotNone(assembler.final_pdf_info[0])
            self.assertEqual(assembler.final_pdf_info[0]["blocks"][0]["content"], "Hello 0")
            self.assertEqual(assembler.final_pdf_info[0]["blocks"][1]["content"], "Hello 1")
            
        asyncio.run(run_test())


class TestAsyncPipelineFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('pipeline.orchestrator.LayoutPredictor')
    @patch('pipeline.orchestrator.ocr_single_image')
    def test_run_pipeline_flow_async(self, mock_ocr, mock_predictor_cls):
        # 1. 模拟 OCR 返回值
        mock_ocr.return_value = "Mocked OCR Result Text"
        
        # 2. 模拟 predictor 实例及 model config
        mock_predictor = mock_predictor_cls.return_value
        from unittest.mock import MagicMock
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "text"}
        mock_predictor.model = mock_model
        
        # 模拟检测结果：1个 text 类型的框
        mock_predictor.predict.return_value = [{
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]), # 映射为 text
            "boxes": torch.tensor([[5.0, 5.0, 150.0, 80.0]])
        }]
        
        # 3. 创建空白临时图片
        img_path = Path(self.temp_dir) / "page_00001.png"
        with Image.new("RGB", (300, 300), (255, 255, 255)) as img:
            img.save(img_path)
            
        from pipeline.orchestrator import run_pipeline_flow_async
        
        async def run_flow():
            pdf_info = await run_pipeline_flow_async(
                img_files=[img_path],
                output_dir=self.temp_dir,
                stem="test_doc",
                table_as_image=True,
                formula_as_image=False,
                keep_header_footer=False,
                max_layout_workers=2,
                ocr_concurrency=4
            )
            return pdf_info
            
        pdf_info = asyncio.run(run_flow())
        
        self.assertIsInstance(pdf_info, list)
        self.assertEqual(len(pdf_info), 1)
        self.assertEqual(pdf_info[0]["page_idx"], 0)
        self.assertEqual(len(pdf_info[0]["blocks"]), 1)
        self.assertEqual(pdf_info[0]["blocks"][0]["content"], "Mocked OCR Result Text")
        
        # 验证 Markdown 文件正确追加写盘
        output_md = Path(self.temp_dir) / "final_output.md"
        self.assertTrue(output_md.exists())
        self.assertIn("Mocked OCR Result Text", output_md.read_text(encoding="utf-8"))



