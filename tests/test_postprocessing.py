import unittest
import sys
from pathlib import Path

# 将父目录加入 sys.path 以便加载 postprocessing 模块
sys.path.append(str(Path(__file__).parent.parent))

from postprocessing import (
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

if __name__ == "__main__":
    unittest.main()
