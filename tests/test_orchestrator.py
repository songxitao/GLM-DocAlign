import os
import shutil
import unittest
from PIL import Image
from unittest.mock import patch
from pipeline.orchestrator import run_pipeline_flow

class TestOrchestrator(unittest.TestCase):
    def test_full_pipeline_orchestration(self):
        # 创建虚拟工作目录与一张模拟 PDF 渲染出的测试图
        test_dir = "tests/temp_run"
        os.makedirs(test_dir, exist_ok=True)
        img_path = os.path.join(test_dir, "page_00001.png")
        
        # 创建一个 600x800 的白底图，画几条代表性的线（模拟倾斜文档和元素）
        img = Image.new("RGB", (600, 800), (255, 255, 255))
        img.save(img_path)
        
        # 模拟 vLLM 返回
        mock_ocr_return = ["Recognized Paragraph Content", "Recognized Table Content"]
        
        # Mock 掉 run_async_ocr 异步 HTTP 调用，而实测本地纠偏、PP-DocLayout检测、XY-Cut排序、裁切去噪的完整流程
        with patch("pipeline.orchestrator.run_async_ocr", return_value=mock_ocr_return) as mock_ocr:
            # 1. 测试 table_as_image=False, formula_as_image=False (使用 OCR 识别)
            md_result, page_middle_data = run_pipeline_flow(
                img_path, 
                test_dir, 
                page_idx=0,
                table_as_image=False,
                formula_as_image=False
            )
            
            # 验证返回内容与中介数据结构
            self.assertTrue(len(md_result) > 0)
            self.assertEqual(page_middle_data["page_idx"], 0)
            self.assertEqual(page_middle_data["page_size"], [600, 800])
            self.assertIn("blocks", page_middle_data)
            
            # 2. 测试 table_as_image=True, formula_as_image=True (物理截图)
            md_result_img, page_middle_data_img = run_pipeline_flow(
                img_path, 
                test_dir, 
                page_idx=1,
                table_as_image=True,
                formula_as_image=True
            )
            self.assertTrue(len(md_result_img) > 0)
            self.assertEqual(page_middle_data_img["page_idx"], 1)
            
        # 清理测试目录
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main()
