# 后处理文本清洗与全半角转换实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完善后处理的高级文本清洗与全半角转换，包括单元测试的编写、测试失败验证、实现核心功能、测试通过验证以及 git 提交。

**Architecture:** 
1. 在 `postprocessing.py` 中定义 `full_to_half_exclude_marks` 函数，只针对大写字母（`FF21-FF3A`）、小写字母（`FF41-FF5A`）以及数字（`FF10-FF19`）进行全角转半角，其他数学符号和特殊字符均保留。
2. 注册 `LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"`，重构 `smart_reflow_markdown` 连字符拼合逻辑，将其移动到最先匹配，防止被英文合并逻辑（Case A）截胡。同时支持去除行末多个连续的特殊连字符。
3. 在 `tests/test_postprocessing.py` 中实现单元测试，并先运行验证失败（TDD 流程），实现后验证通过，并完成 Git 提交。

**Tech Stack:** Python 3.x, unittest, Git

---

### Task 1: 编写单元测试

**Files:**
- Create: `E:/project/GLM-OCR/tests/test_postprocessing.py`

- [ ] **Step 1: 创建测试文件并编写测试用例**

```python
import unittest
import sys
from pathlib import Path

# 将父目录加入 sys.path 以加载 postprocessing
sys.path.append(str(Path(__file__).parent.parent))

from postprocessing import (
    full_to_half_exclude_marks,
    smart_reflow_markdown
)

class TestPostProcessing(unittest.TestCase):
    def test_full_to_half_exclude_marks(self):
        # 验证全角英文字母、数字规范转换为半角，而保留特殊公式符号（如 ＋, ＝, ！ 等）
        text = "Ｔｅｓｔ １２３！ ＋＝ ａｂｃ"
        expected = "Test 123！ ＋＝ abc"
        self.assertEqual(full_to_half_exclude_marks(text), expected)

    def test_smart_reflow_with_special_hyphen(self):
        # 验证对包含特殊断词连字符（例如 \u00ad）的行末断词进行智能合并
        text = "infor-\u00ad\nmation"
        expected = "information"
        self.assertEqual(smart_reflow_markdown(text), expected)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证其执行并失败**

运行命令：
```powershell
python -m unittest tests/test_postprocessing.py
```
预期结果：
失败（FAIL），错误信息提示 `ImportError: cannot import name 'full_to_half_exclude_marks'`。

---

### Task 2: 实现功能并使测试通过

**Files:**
- Modify: `E:/project/GLM-OCR/postprocessing.py`

- [ ] **Step 1: 在 `postprocessing.py` 中增加并修改对应逻辑**

在 `postprocessing.py` 中实现：
1. `full_to_half_exclude_marks(text: str) -> str`
2. 注册 `LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"` 并在 `smart_reflow_markdown` 中调整优先级，优先合并连字符并清理行尾多余的连字符。

```python
# 注册连字符字符集
LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"

def full_to_half_exclude_marks(text: str) -> str:
    """
    仅转换 FF21-FF3A (全角大写 A-Z), FF41-FF5A (全角小写 a-z), FF10-FF19 (全角数字 0-9) 范围内的字符。
    其他字符（如公式符号 ＋, ＝，标点 ！）保持原样。
    """
    res = []
    for char in text:
        code = ord(char)
        # 全角数字 FF10-FF19, 全角大写 FF21-FF3A, 全角小写 FF41-FF5A
        if (0xFF10 <= code <= 0xFF19) or (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A):
            res.append(chr(code - 0xfee0))
        else:
            res.append(char)
    return "".join(res)
```

修改 `smart_reflow_markdown` 中的连字符匹配逻辑（将 Case B 优先级提升至 Case A 之前）：

```python
            # 情况 B: 连字符合并 (如 infor-\u00ad\nmation)
            if prev_char in LINE_END_HYPHEN_CHARS:
                should_merge = True
                merge_sep = ""
                # 去除行尾所有的连字符字符，同时保留前导空白
                stripped_buf = buffer.strip()
                while stripped_buf and stripped_buf[-1] in LINE_END_HYPHEN_CHARS:
                    stripped_buf = stripped_buf[:-1]
                leading_spaces = buffer[:len(buffer) - len(buffer.lstrip())]
                buffer = leading_spaces + stripped_buf

            # 情况 A: 英文合并 (非句号结尾 + 小写开头)
            elif not is_sentence_end(prev_char) and curr_char.islower():
                should_merge = True
```

同时，在 `stitch_pages` 中，也应用类似的连字符拼合逻辑：

```python
        # 1. 英文连字符
        if prev_end in LINE_END_HYPHEN_CHARS:
            # 去除前一页末尾所有的连字符
            stripped_text = final_text.strip()
            while stripped_text and stripped_text[-1] in LINE_END_HYPHEN_CHARS:
                stripped_text = stripped_text[:-1]
            final_text = stripped_text + page_text.strip()
```

- [ ] **Step 2: 再次运行测试验证其通过**

运行命令：
```powershell
python -m unittest tests/test_postprocessing.py
```
预期结果：
测试通过（OK）。

---

### Task 3: Git 提交

- [ ] **Step 1: 添加并提交代码**

运行命令：
```powershell
git add postprocessing.py tests/test_postprocessing.py
git commit -m "feat: implement full-to-half conversion and advanced hyphenation reflow"
```
预期结果：
代码成功提交至 Git 仓库。
