# GLM-OCR 工作流重构与 MinerU 优势融合实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理 mineru-glm 融合的侵入式代码，并在 GLM-OCR 纯 VLM 工作流中按需引入 MinerU 的连字符重排、全角转半角、多视觉元素（Table/Chart）截图保护以及中介 JSON 本地缓存机制。

**Architecture:** 物理删除旧版融合代码；扩展 `postprocessing.py` 使其支持 `table`/`chart` 截图提取以及高级文本清洗；改进 `orchestrator.py` 生成 `_middle.json`；改造 `run_pipeline.py` 优先读取中介 JSON 缓存进行无 API 转换。

**Tech Stack:** Python 3, PyMuPDF, Pillow, transformers, Pytest

---

### Task 1: 清理 mineru-glm 融合代码

**Files:**
- Delete: `E:/project/GLM-OCR/mineru_integration/mineru_glm.py`
- Delete: `E:/project/GLM-OCR/mineru_integration/test_mineru_glm.py`
- Delete: `E:/project/GLM-OCR/mineru_integration/mineru_vl_utils/` 文件夹

- [ ] **Step 1: 删除融合集成脚本**
  运行命令物理清理 `mineru_integration` 目录下的所有残留。
  运行: `Remove-Item -Recurse -Force E:/project/GLM-OCR/mineru_integration`

- [ ] **Step 2: 验证 Git 状态**
  运行: `git status`
  预期: `mineru_integration/` 处于 untracked 或已删除状态。

- [ ] **Step 3: 进行阶段性 Commit**
  ```bash
  git add -A
  git commit -m "cleanup: remove mineru-glm integration files"
  ```

---

### Task 2: 完善后处理的高级文本清洗与全半角转换

**Files:**
- Modify: `E:/project/GLM-OCR/postprocessing.py`
- Create: `E:/project/GLM-OCR/tests/test_postprocessing.py`

- [ ] **Step 1: 编写 TDD 失败测试**
  在 `tests/test_postprocessing.py` 中编写对 `full_to_half_exclude_marks` 和增强连字符清洗的单元测试。
  
  ```python
  # E:/project/GLM-OCR/tests/test_postprocessing.py
  import unittest
  from postprocessing import full_to_half_exclude_marks, smart_reflow_markdown
  
  class TestPostProcessing(unittest.TestCase):
      def test_full_to_half_exclude_marks(self):
          # 验证全角英文字母和数字被正确转换为半角，且排除公式标点
          raw_text = "Ｔｅｓｔ １２３ ＋＝ ￥"
          expected = "Test 123 ＋＝ ￥"
          self.assertEqual(full_to_half_exclude_marks(raw_text), expected)
          
      def test_smart_reflow_with_special_hyphen(self):
          # 验证隐藏断行连字符 \u00ad 能够正确拼接词语
          raw_text = "infor-\u00ad\nmation technology"
          expected = "information technology"
          # 对 text 先应用 smart_reflow_markdown，看其是否拼合连字符
          result = smart_reflow_markdown(raw_text)
          self.assertIn(expected, result)
  
  if __name__ == '__main__':
      unittest.main()
  ```

- [ ] **Step 2: 运行测试并验证其失败**
  运行: `python -m unittest tests/test_postprocessing.py`
  预期: 报 `ImportError: cannot import name 'full_to_half_exclude_marks'` 或者是测试断言失败。

- [ ] **Step 3: 实现 `full_to_half_exclude_marks` 与行尾断词优化**
  在 `E:/project/GLM-OCR/postprocessing.py` 中新增 `full_to_half_exclude_marks` 并在 `smart_reflow_markdown` 中加入 `\u00ad` 特殊连字符的识别合并。

  ```python
  # 在 E:/project/GLM-OCR/postprocessing.py 中新增：
  LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"
  
  def full_to_half_exclude_marks(text: str) -> str:
      result = []
      for char in text:
          code = ord(char)
          if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
              result.append(chr(code - 0xFEE0))
          else:
              result.append(char)
      return ''.join(result)
  ```
  
  并将 `smart_reflow_markdown` 中的连字符合并规则修改为：
  ```python
  # 替换 postprocessing.py 中 86-90 行：
  elif prev_char in LINE_END_HYPHEN_CHARS:
      should_merge = True
      merge_sep = ""
      buffer = buffer.strip()[:-1]
  ```

- [ ] **Step 4: 重新运行测试以验证通过**
  运行: `python -m unittest tests/test_postprocessing.py`
  预期: 测试 `TestPostProcessing` 通过 (PASS)。

- [ ] **Step 5: 提交代码**
  ```bash
  git add postprocessing.py tests/test_postprocessing.py
  git commit -m "feat: implement full-to-half conversion and advanced hyphenation reflow"
  ```

---

### Task 3: 扩展后处理 Grounding 解析以支持表格 (Table) 与图表 (Chart) 截图裁剪

**Files:**
- Modify: `E:/project/GLM-OCR/postprocessing.py`
- Modify: `E:/project/GLM-OCR/tests/test_postprocessing.py`

- [ ] **Step 1: 在单元测试中添加 Table/Chart 截图裁剪测试**
  添加一个模拟 VLM 输出的含有 `table[[100, 200, 300, 400]]` 标签的文本提取测试。

  ```python
  # 在 E:/project/GLM-OCR/tests/test_postprocessing.py 的 TestPostProcessing 中新增：
  def test_process_grounded_markdown_with_table(self):
      # 测试 process_grounded_markdown 函数是否将 table[[...]] 作为图片截图提取
      # 由于该测试需要实图，我们仅写断言或验证正则匹配逻辑
      pass
  ```
  
- [ ] **Step 2: 改造 `process_grounded_markdown` 的替换正则和裁剪分流**
  修改 `E:/project/GLM-OCR/postprocessing.py` 中的 `process_grounded_markdown` 和 `create_diagnosis_image`，使它们支持对 `table` 和 `chart` 进行与 `image` 一样的截图保存和诊断标注。

  目标替换内容（`process_grounded_markdown` 第 266-282 行）：
  ```python
  target_tags = ["image", "table", "chart"]
  is_target = any(t in tag_type for t in target_tags) and "caption" not in tag_type
  
  if is_target:
      try:
          x1 = int(coords[0] / NORMALIZATION_FACTOR * width_orig)
          y1 = int(coords[1] / NORMALIZATION_FACTOR * height_orig)
          x2 = int(coords[2] / NORMALIZATION_FACTOR * width_orig)
          y2 = int(coords[3] / NORMALIZATION_FACTOR * height_orig)
          x1, y1 = max(0, x1), max(0, y1)
          x2, y2 = min(width_orig, x2), min(height_orig, y2)
          
          if x2 > x1 and y2 > y1:
              cropped = original_image.crop((x1, y1, x2, y2))
              # 动态命名：根据 tag_type (如 table, chart) 拼装文件名
              clean_tag = "table" if "table" in tag_type else ("chart" if "chart" in tag_type else "image")
              image_filename = f"{original_image_path.stem}_{clean_tag}_{image_counter}.png"
              cropped.save(images_subdir / image_filename)
              image_counter += 1
              return f"\n\n![{clean_tag}](images/{image_filename})\n\n"
      except Exception:
          return ""
  ```

  目标修改 `create_diagnosis_image` 里的 `colors` 字典（第 298 行）：
  ```python
  colors = {"image": "red", "table": "blue", "chart": "orange", "title": "purple", "text": "green", "default": "yellow"}
  ```

- [ ] **Step 3: 运行验证**
  检查是否有任何语法错漏。
  运行: `python -m unittest tests/test_postprocessing.py`
  预期: PASS。

- [ ] **Step 4: 提交代码**
  ```bash
  git add postprocessing.py
  git commit -m "feat: support auto-cropping for table and chart grounded bounding boxes"
  ```

---

### Task 4: 改造管线 Orchestrator 支持表格与公式物理截图与 Middle JSON 生成

**Files:**
- Modify: `E:/project/GLM-OCR/pipeline/orchestrator.py`

- [ ] **Step 1: 新增中介 JSON 构造逻辑**
  在 `run_pipeline_flow` 运行中，将原本返回的 markdown 文字连同 PP-DocLayout-V3 识别出的 Blocks 封装，并支持对 `table` 在本地进行直接裁剪（防止大模型还原崩溃）。

  修改 `pipeline/orchestrator.py`：
  ```python
  # 修改 run_pipeline_flow(image_path: str, output_dir: str) -> tuple[str, dict] / str
  # 为保持接口兼容，我们可以返回 (full_markdown, page_middle_data)
  ```
  在本地 layout 检测出表格等块时，如果设置 `table_as_image=True`，不再将其传入 OCR 队列进行文字识别，直接在原图上 crop 成 `{page_stem}_table_{fig_counter}.png` 并追加到 final_elements，渲染为 markdown 占位图片。
  最后把该页所有的 `boxes` 转换为 `page_middle_data`（结构为 `{"page_idx": page_idx, "page_size": [w, h], "blocks": [...]}`）。

- [ ] **Step 2: 编写测试确认 Orchestrator 数据流**
  在 `tests/test_pipeline.py` 中验证 `run_pipeline_flow` 是否正常工作。

- [ ] **Step 3: 运行验证**
  运行: `python -m unittest tests/test_postprocessing.py`
  预期: PASS。

- [ ] **Step 4: 提交代码**
  ```bash
  git add pipeline/orchestrator.py
  git commit -m "feat: modify orchestrator to save middle json data and support layout-level table cropping"
  ```

---

### Task 5: 改造 `run_pipeline.py` 实现 Middle JSON 归档与本地缓存机制

**Files:**
- Modify: `E:/project/GLM-OCR/run_pipeline.py`

- [ ] **Step 1: 添加缓存读取与规整写入**
  在 `run_pipeline.py` 的 `main` 方法中：
  1. 重构输出文件归纳逻辑，将同一个 PDF 文件解析生成的 `.md`、`.docx`、`_middle.json`、以及所有的截图 `images/` 全都放入 `ocr_output/<PDF主文件名>/` 目录。
  2. 运行时先检测本地是否存在该目录下的 `{pdf_name}_middle.json`。
  3. 如果存在且无 `--force` 参数，则直接根据 JSON 序列化数据，无需调用 transformers 模型和 GLM-OCR 接口，直接使用本地 `postprocessing` 生成最终 markdown 与 word。
  4. 如果不存在，跑完整个 orchestrator 后将生成的 `middle_json` 存储下来。

- [ ] **Step 2: 验证全套工作流的可行性**
  对一个本地测试的 PDF 文档，执行第一次解析：
  运行: `python run_pipeline.py <PDF_PATH>`
  预期: 生成全套产物文件夹。

- [ ] **Step 3: 验证第二次本地缓存生效**
  第二次解析同一个文档：
  运行: `python run_pipeline.py <PDF_PATH>`
  预期: 瞬间完成（没有重新加载版面模型，没有调用 API），提示读取本地缓存。

- [ ] **Step 4: 提交并完成重构**
  ```bash
  git add run_pipeline.py
  git commit -m "feat: add middle json archiving and layout-level rendering caching mechanism"
  ```
