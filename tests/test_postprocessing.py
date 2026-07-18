import unittest
import sys
from pathlib import Path


from glmocr.postprocessing import (
    full_to_half_exclude_marks,
    smart_reflow_markdown
)

class TestPostProcessing(unittest.TestCase):
    def test_full_to_half_exclude_marks(self):
        # 验证将全角英文字母、数字规范转换为半角，而保留特殊公式符号（如 ＋, ＝, ！ 等）
        text = "Ｔｅｓｔ １２３！ ＋＝ ａｂｃ"
        expected = "Test 123！ ＋＝ abc"
        self.assertEqual(full_to_half_exclude_marks(text), expected)

    def test_smart_reflow_with_special_hyphen(self):
        # 验证对包含特殊断词连字符（例如 \u00ad）的行末断词进行智能合并（例如 infor-\u00ad\nmation 拼合为 information）
        text = "infor-\u00ad\nmation"
        expected = "information"
        self.assertEqual(smart_reflow_markdown(text), expected)

    def test_process_grounded_markdown_table_and_chart(self):
        import tempfile
        from PIL import Image
        from glmocr.postprocessing import process_grounded_markdown
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            
            # 创建临时纯白 100x100 图像
            img = Image.new("RGB", (100, 100), "white")
            img_path = tmp_dir_path / "test_page.png"
            img.save(img_path)
            
            # 创建临时 Markdown 文件
            md_path = tmp_dir_path / "test_page.md"
            md_content = (
                "This is a table test: table[[10, 20, 80, 90]]\n"
                "This is a chart test: chart[[5, 5, 50, 50]]\n"
                "This is an image test: image[[15, 15, 60, 60]]\n"
            )
            md_path.write_text(md_content, encoding="utf-8")
            
            # 执行 process_grounded_markdown
            process_grounded_markdown(md_path, img_path, tmp_dir_path)
            
            # 检查裁剪截图文件是否存在
            table_img = tmp_dir_path / "images" / "test_page_table_1.png"
            chart_img = tmp_dir_path / "images" / "test_page_chart_1.png"
            image_img = tmp_dir_path / "images" / "test_page_image_1.png"
            
            self.assertTrue(table_img.exists(), "Table image should be cropped and saved.")
            self.assertTrue(chart_img.exists(), "Chart image should be cropped and saved.")
            self.assertTrue(image_img.exists(), "Image image should be cropped and saved.")
            
            # 检查大小不为0
            self.assertTrue(table_img.stat().st_size > 0)
            self.assertTrue(chart_img.stat().st_size > 0)
            self.assertTrue(image_img.stat().st_size > 0)
            
            # 检查替换后的 Markdown 内容
            processed_md = md_path.read_text(encoding="utf-8")
            self.assertIn("![table](images/test_page_table_1.png)", processed_md)
            self.assertIn("![chart](images/test_page_chart_1.png)", processed_md)
            self.assertIn("![image](images/test_page_image_1.png)", processed_md)

    def test_create_diagnosis_image_colors(self):
        import tempfile
        from PIL import Image
        from glmocr.postprocessing import create_diagnosis_image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            
            # 创建临时纯白 100x100 图像
            img = Image.new("RGB", (100, 100), "white")
            img_path = tmp_dir_path / "test_page.png"
            img.save(img_path)
            
            # 创建临时 Markdown 文件，包含 table, chart, image 等 tags
            md_path = tmp_dir_path / "test_page.md"
            md_content = (
                "table[[10, 20, 80, 90]]\n"
                "chart[[5, 5, 50, 50]]\n"
                "image[[15, 15, 60, 60]]\n"
            )
            md_path.write_text(md_content, encoding="utf-8")
            
            # 执行 create_diagnosis_image
            create_diagnosis_image(md_path, img_path, tmp_dir_path)
            
            # 确认生成了诊断图
            diagnosis_img = tmp_dir_path / "diagnosis" / "test_page_diagnosis.png"
            self.assertTrue(diagnosis_img.exists(), "Diagnosis image should be created.")
            self.assertTrue(diagnosis_img.stat().st_size > 0)

if __name__ == "__main__":
    unittest.main()

