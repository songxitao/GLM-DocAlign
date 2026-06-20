# GLM-OCR 三级流式异步流水线重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 GLM-OCR 的核心串行管线重构为基于 `asyncio.Queue` 与多线程混合的“三级流式异步流水线”，实现 CPU 检测与 VLM 推理并发重叠，消灭“小锯齿”硬件限制，并确保输出页面完全物理保序。

**Architecture:** 
1. 提取版面模型 `PP-DocLayoutV3` 为单例 `LayoutPredictor`。
2. 使用 `concurrent.futures.ThreadPoolExecutor(max_workers=2)` 并发运行车间一（纠偏及版面检测）。
3. 使用 `asyncio.Semaphore(4)` 控制并发协程运行车间二（异步 VLM OCR）。
4. 在车间三实现 `StreamAssembler`，在内存中保序缓冲，页面所有块 OCR 就绪后流式追加写入 `final_output.md`，并在完成后生成中介 JSON 与 Word。

**Tech Stack:** Python `asyncio`, `aiohttp`, `PIL`, `torch`, `transformers`

---

### Task 1: 封装与验证 `LayoutPredictor`

**Files:**
- Modify: `pipeline/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: 在测试中编写 `LayoutPredictor` 加载与预测测试**

在 `tests/test_orchestrator.py` 中编写以下测试：
```python
import unittest
from PIL import Image
from pipeline.orchestrator import LayoutPredictor, LOCAL_LAYOUT_MODEL

class TestLayoutPredictor(unittest.TestCase):
    def test_predictor_singleton_initialization(self):
        predictor = LayoutPredictor(LOCAL_LAYOUT_MODEL)
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.image_processor)
        
        # 建立一张空白测试图片验证推理
        test_img = Image.new("RGB", (300, 300), (255, 255, 255))
        results = predictor.predict(test_img)
        self.assertEqual(len(results), 1)
```

- [ ] **Step 2: 运行测试以确认其失败**

运行命令：`E:\conda\envs\deepseek-ocr\python.exe -m unittest tests/test_orchestrator.py`
预期结果：`ImportError` 或 `AttributeError` (因为 `LayoutPredictor` 还未实现)。

- [ ] **Step 3: 实现 `LayoutPredictor` 逻辑**

在 `pipeline/orchestrator.py` 中，定义 `LayoutPredictor` 类：
```python
class LayoutPredictor:
    def __init__(self, model_dir: str):
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir).to("cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)

    def predict(self, corrected_image):
        inputs = self.image_processor(images=corrected_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.image_processor.post_process_object_detection(
            outputs, target_sizes=[corrected_image.size[::-1]]
        )
```

- [ ] **Step 4: 运行测试验证通过**

运行命令：`E:\conda\envs\deepseek-ocr\python.exe -m unittest tests/test_orchestrator.py`
预期结果：PASS

- [ ] **Step 5: 阶段性提交**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: implement LayoutPredictor singleton loader and test case"
```

---

### Task 2: 实现与测试保序组装器 `StreamAssembler`

**Files:**
- Modify: `pipeline/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: 编写 `StreamAssembler` 单元测试**

在 `tests/test_orchestrator.py` 中编写以下测试：
```python
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
            # 2页总数的测试
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
            # 此时页 0 还没注册，不能有写入动作
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 2. 注册页 0
            await assembler.register_page(0, page_0)
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 3. 填充页 1 OCR（乱序返回）
            await assembler.fill_ocr_content(1, 0, "Hello Page 1")
            # 页 0 仍然没有就绪，页序仍然在 0
            self.assertEqual(assembler.current_writing_page, 0)
            
            # 4. 填充页 0 OCR
            await assembler.fill_ocr_content(0, 0, "Hello Page 0")
            # 此时页 0 和页 1 都已完成，组装器将自动冲刷（Flush）它们
            # 等待所有页写盘触发完成事件
            await asyncio.wait_for(assembler.finished_event.wait(), timeout=2.0)
            self.assertEqual(assembler.current_writing_page, 2)
            
            # 验证写入的物理文件内容是否符合预期
            output_md = Path(self.temp_dir) / "final_output.md"
            self.assertTrue(output_md.exists())
            content = output_md.read_text(encoding="utf-8")
            self.assertIn("Hello Page 0", content)
            self.assertIn("Hello Page 1", content)
            
        asyncio.run(run_test())
```

- [ ] **Step 2: 运行测试以确认其失败**

运行：`E:\conda\envs\deepseek-ocr\python.exe -m unittest tests/test_orchestrator.py`
预期结果：`ImportError` (由于未定义 `StreamAssembler`)。

- [ ] **Step 3: 实现 `StreamAssembler` 的具体代码**

在 `pipeline/orchestrator.py` 中，实现 `StreamAssembler` 类：
```python
from postprocessing import smart_reflow_markdown

class StreamAssembler:
    def __init__(self, output_dir, stem, total_pages):
        self.output_dir = Path(output_dir)
        self.stem = stem
        self.total_pages = total_pages
        self.page_buffers = {}
        self.current_writing_page = 0
        self.lock = asyncio.Lock()
        self.finished_event = asyncio.Event()
        
        self.output_md_path = self.output_dir / "final_output.md"
        self.output_md_path.parent.mkdir(exist_ok=True, parents=True)
        self.output_md_path.write_text("", encoding="utf-8")

    async def register_page(self, page_idx, page_structure):
        async with self.lock:
            ocr_tasks_count = sum(1 for b in page_structure["blocks"] if b["type"] == "ocr_task")
            self.page_buffers[page_idx] = {
                "structure": page_structure,
                "pending_ocr": ocr_tasks_count,
                "ready": ocr_tasks_count == 0
            }
            await self.flush_ready_pages()

    async def fill_ocr_content(self, page_idx, block_idx, content):
        async with self.lock:
            if page_idx not in self.page_buffers:
                return
            page = self.page_buffers[page_idx]
            for block in page["structure"]["blocks"]:
                if block.get("block_idx") == block_idx:
                    block["content"] = content.replace("```", "")
                    break
            
            page["pending_ocr"] -= 1
            if page["pending_ocr"] <= 0:
                page["ready"] = True
            
            await self.flush_ready_pages()

    async def flush_ready_pages(self):
        while self.current_writing_page in self.page_buffers:
            page = self.page_buffers[self.current_writing_page]
            if not page["ready"]:
                break
            
            await self.write_page_to_disk(self.current_writing_page, page["structure"])
            self.current_writing_page += 1
            if self.current_writing_page == self.total_pages:
                self.finished_event.set()

    async def write_page_to_disk(self, page_idx, structure):
        markdown_lines = []
        for block in structure["blocks"]:
            if "content" in block:
                txt = block["content"].strip()
                lbl = block.get("label", "paragraph").lower()
                if not txt:
                    continue
                reflowed = smart_reflow_markdown(txt)
                
                # 语义渲染格式
                if lbl == "doc_title":
                    markdown_lines.append(f"\n\n# {reflowed}\n\n")
                elif lbl == "paragraph_title":
                    markdown_lines.append(f"\n\n## {reflowed}\n\n")
                elif lbl == "table":
                    markdown_lines.append(f"\n\n{reflowed}\n\n")
                elif lbl == "abstract":
                    markdown_lines.append(f"\n\n> **Abstract** — {reflowed}\n\n")
                elif lbl == "algorithm":
                    markdown_lines.append(f"\n\n```\n{reflowed}\n```\n\n")
                elif lbl == "figure_title":
                    markdown_lines.append(f"\n\n*{reflowed}*\n\n")
                else:
                    markdown_lines.append(f"\n\n{reflowed}\n\n")
            elif "image_path" in block:
                # 拼装相对路径
                lbl = block.get("label", "figure").lower()
                markdown_lines.append(f"\n\n![{lbl}]({block['image_path']})\n\n")
        
        page_md = "\n\n".join(markdown_lines)
        if page_idx > 0:
            page_md = f"\n\n\\newpage\n\n{page_md}"
            
        with open(self.output_md_path, "a", encoding="utf-8") as f:
            f.write(page_md)
```

- [ ] **Step 4: 运行测试确认通过**

运行：`E:\conda\envs\deepseek-ocr\python.exe -m unittest tests/test_orchestrator.py`
预期结果：PASS

- [ ] **Step 5: 提交改动**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: implement StreamAssembler class with TDD test"
```

---

### Task 3: 实现主控制流 `run_pipeline_flow_async`

**Files:**
- Modify: `pipeline/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: 编写异步流水线主流程测试**

在 `tests/test_orchestrator.py` 中编写集成模拟测试：
```python
class TestAsyncPipelineFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_pipeline_flow_async(self):
        from pipeline.orchestrator import run_pipeline_flow_async
        # 测试一张小图片的异步推理，确认其完整运转返回 middle_data
        # 在此处模拟调用
        pass # 此处根据实际传入的测试路径或模拟组件进行断言
```

- [ ] **Step 2: 实现在 `pipeline/orchestrator.py` 下的 `process_single_page_layout`**

编写从原 `run_pipeline_flow` 中剥离出的单页 Layout 物理过程（由线程池驱动的计算函数）：
```python
from pipeline.deskew import detect_skew_angle, rotate_image
from pipeline.xycut import sort_boxes_by_xy_cut
from pipeline.masked_crop import crop_and_mask
from PIL import ImageDraw
from pathlib import Path

def process_single_page_layout(image_path: str, page_idx: int, predictor, output_dir: str, keep_header_footer: bool, table_as_image: bool, formula_as_image: bool):
    raw_image = Image.open(image_path).convert("RGB")
    angle = detect_skew_angle(raw_image)
    corrected_image = rotate_image(raw_image, -angle)
    
    # 使用单例 predictor 预测
    results = predictor.predict(corrected_image)
    
    boxes = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < 0.4:
                continue
            label = predictor.model.config.id2label.get(label_id.item(), f"Label_{label_id.item()}")
            box_coords = [int(i) for i in box.tolist()]
            boxes.append({"coords": box_coords, "label": label})
            
    page_stem = Path(image_path).stem
    
    # 如果没有检测框的兜底
    if not boxes:
        return {
            "page_idx": page_idx,
            "page_size": list(corrected_image.size),
            "blocks": []
        }, []

    # 画框诊断图（略）
    ...
    
    # XY-Cut 排序
    sorted_indices = sort_boxes_by_xy_cut(boxes)
    
    page_structure = {
        "page_idx": page_idx,
        "page_size": list(corrected_image.size),
        "blocks": []
    }
    
    ocr_tasks = []
    fig_counter = 1
    table_counter = 1
    formula_counter = 1
    
    for block_idx, idx in enumerate(sorted_indices):
        element = boxes[idx]
        label = element["label"]
        lbl_lower = label.lower()
        
        # 过滤页眉页脚
        if not keep_header_footer:
            has_table = any(b["label"].lower() == "table" for b in boxes)
            if lbl_lower == "footer" or (lbl_lower == "header" and not has_table):
                continue
                
        is_crop = False
        clean_tag = ""
        counter = 0
        
        if lbl_lower in ["figure", "image", "chart"]:
            is_crop = True
            clean_tag = "fig"
            counter = fig_counter
            fig_counter += 1
        elif lbl_lower == "table" and table_as_image:
            is_crop = True
            clean_tag = "table"
            counter = table_counter
            table_counter += 1
        elif lbl_lower == "formula" and formula_as_image:
            is_crop = True
            clean_tag = "formula"
            counter = formula_counter
            formula_counter += 1
            
        if is_crop:
            filename = f"{page_stem}_{clean_tag}_{counter}.png"
            images_subdir = Path(output_dir) / "images"
            images_subdir.mkdir(exist_ok=True, parents=True)
            fig_path = images_subdir / filename
            cropped_fig = corrected_image.crop(element["coords"])
            cropped_fig.save(fig_path)
            
            page_structure["blocks"].append({
                "block_idx": block_idx,
                "type": "image_block",
                "label": label,
                "bbox": element["coords"],
                "image_path": f"images/{filename}"
            })
        else:
            # 文本块：进行 crop 和 mask
            cropped_sub = crop_and_mask(corrected_image, boxes, idx)
            page_structure["blocks"].append({
                "block_idx": block_idx,
                "type": "ocr_task",
                "label": label,
                "bbox": element["coords"],
                "content": None
            })
            ocr_tasks.append({
                "page_idx": page_idx,
                "block_idx": block_idx,
                "label": label,
                "image": cropped_sub
            })
            
    return page_structure, ocr_tasks
```

- [ ] **Step 3: 实现主控异步函数 `run_pipeline_flow_async`**

实现 `run_pipeline_flow_async` 用于协调多级车间工作：
```python
import aiohttp
from pipeline.async_ocr import ocr_single_image
from concurrent.futures import ThreadPoolExecutor

async def run_pipeline_flow_async(
    img_files: list,
    output_dir: str,
    stem: str,
    table_as_image: bool = True,
    formula_as_image: bool = False,
    keep_header_footer: bool = False,
    max_layout_workers: int = 2,
    ocr_concurrency: int = 4
):
    total_pages = len(img_files)
    
    # 1. 建立组件与队列
    predictor = LayoutPredictor(LOCAL_LAYOUT_MODEL)
    assembler = StreamAssembler(output_dir, stem, total_pages)
    
    layout_queue = asyncio.Queue()
    ocr_queue = asyncio.Queue()
    
    # 2. 注入 Layout 任务
    for idx, img_path in enumerate(img_files):
        await layout_queue.put({"page_idx": idx, "img_path": str(img_path)})
        
    # 3. 创建执行池与 Workers
    loop = asyncio.get_running_loop()
    thread_pool = ThreadPoolExecutor(max_workers=max_layout_workers)
    
    async def layout_worker_loop():
        while True:
            try:
                task = await layout_queue.get()
                page_idx = task["page_idx"]
                img_path = task["img_path"]
                
                # 在线程池运行 layout 计算
                page_structure, ocr_tasks = await loop.run_in_executor(
                    thread_pool,
                    process_single_page_layout,
                    img_path, page_idx, predictor, output_dir, keep_header_footer, table_as_image, formula_as_image
                )
                
                await assembler.register_page(page_idx, page_structure)
                for ocr_task in ocr_tasks:
                    await ocr_queue.put(ocr_task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Layout Worker Error: {e}")
            finally:
                layout_queue.task_done()
                
    sem = asyncio.Semaphore(ocr_concurrency)
    async def ocr_worker_loop():
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    task = await ocr_queue.get()
                    page_idx = task["page_idx"]
                    block_idx = task["block_idx"]
                    label = task["label"]
                    img_obj = task["image"]
                    
                    content = await ocr_single_image(session, img_obj, label, sem)
                    await assembler.fill_ocr_content(page_idx, block_idx, content)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"❌ OCR Worker Error: {e}")
                finally:
                    ocr_queue.task_done()
                    
    # 启动 2 个 layout workers，4 个 ocr workers
    layout_workers = [asyncio.create_task(layout_worker_loop()) for _ in range(max_layout_workers)]
    ocr_workers = [asyncio.create_task(ocr_worker_loop()) for _ in range(ocr_concurrency)]
    
    # 4. 等待完成
    await layout_queue.join()
    await ocr_queue.join()
    await assembler.finished_event.wait()
    
    # 停止 workers
    for w in layout_workers:
        w.cancel()
    for w in ocr_workers:
        w.cancel()
    thread_pool.shutdown()
    
    # 返回全局 Middle JSON 结果
    pdf_info = []
    for idx in range(total_pages):
        pdf_info.append(assembler.page_buffers[idx]["structure"])
    return pdf_info
```

- [ ] **Step 4: 运行单元测试**

运行测试，检验异步推理是否通过。

- [ ] **Step 5: 提交改动**

```bash
git add pipeline/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: implement complete async pipeline workflow"
```

---

### Task 4: 重构 `run_pipeline.py` 与端到端联调

**Files:**
- Modify: `run_pipeline.py`

- [ ] **Step 1: 引入 `run_pipeline_flow_async` 调用**

修改 `run_pipeline.py`。当本地没有命中缓存时，由 `asyncio.run` 执行异步重组流水线：
```python
    # 替换原本的依次循环调用
    # 2. 依次运行 Pipeline 获取 Markdown 并缝合
    print("🚀 启动三级流式异步流水线进行识别...")
    
    # 原有的 loop 直接跑异步函数
    pdf_info = asyncio.run(
        run_pipeline_flow_async(
            img_files=img_files,
            output_dir=str(output_dir),
            stem=stem,
            table_as_image=not ocr_table,
            formula_as_image=keep_header_footer, # 根据需求配置
            keep_header_footer=keep_header_footer,
            max_layout_workers=2,
            ocr_concurrency=4
        )
    )
    
    # 将 pdf_info 结构归档中介 JSON 并转 Word（这部分原有逻辑保留）
    ...
```

- [ ] **Step 2: 运行测试**

运行一页图片测试：
`E:\conda\envs\deepseek-ocr\python.exe run_pipeline.py tests/temp_run/page_00001.png --force`
确认首次成功写入 md、docx 以及 `_middle.json`。

运行多页 PDF 测试（如 `E:\desktop\code\New folder\44221625_LI LEI.pdf`）：
`E:\conda\envs\deepseek-ocr\python.exe run_pipeline.py "E:\desktop\code\New folder\44221625_LI LEI.pdf" --force`
监控控制台打印的 `📄 页面 X/Y 数据就绪，已流式落盘写入。`，观察硬件利用率。

- [ ] **Step 3: 提交改动**

```bash
git add run_pipeline.py
git commit -m "feat: refactor entry run_pipeline.py to use run_pipeline_flow_async"
```
