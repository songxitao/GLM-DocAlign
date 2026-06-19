# GLM-OCR 三级流式异步流水线重构设计规范

## 1. 目标与背景
多页 PDF 文件在进行 OCR 识别时，原先的串行架构（处理完一页再处理下一页，且每页重复加载版面模型）存在硬件利用率低、CPU与GPU交替闲置（“小锯齿”现象）、内存因模型频繁加载卸载而抖动的问题。
本设计旨在重构 GLM-OCR 的核心运行管线，实现“三级流式异步流水线”。在 Windows 平台上，控制 CPU 密集型任务并发以防卡顿，同时提升 GPU 调用饱和度，并确保输出文档页序物理一致。

## 2. 系统架构与并发限制
系统基于 Python `asyncio` 协程与 `concurrent.futures.ThreadPoolExecutor` 混合驱动。

```
                    ┌────────────────────────┐
                    │     PDF / 图片源输入    │
                    └───────────┬────────────┘
                                │ 页面拆分
                                ▼
                       [ layout_queue ]
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │ 车间一: Layout检测 & 纠偏 (2 线程池)          │
         │ - PP-DocLayoutV3 模型长驻单例                 │
         │ - 图片纠偏、画框诊断图                         │
         │ - 直接截图块裁剪 (Image Block)                 │
         └──────────────┬───────────────────────────────┘
                        │                           │
         注册页面结构    │                           │ 推送待OCR子图任务
         (Page Skeleton)│                           │ (OCR Task)
                        ▼                           ▼
            ┌──────────────────────┐        [ ocr_queue ]
            │                      │                │
            │                      │                ▼
            │                      │    ┌──────────────────────┐
            │   车间三: 组装拼写   │◄───┤  车间二: VLM OCR 推理  │
            │   (StreamAssembler)  │    │  (4 协程并发 I/O)    │
            │                      │    └──────────────────────┘
            └───────────┬──────────┘
                        │
                        ▼
       ┌──────────────────────────────────┐
       │   流式写盘 md -> word -> json     │
       └──────────────────────────────────┘
```

### 并发限制指标
*   **车间一 (Layout 线程池)**：最大并发 **2**。使用 `concurrent.futures.ThreadPoolExecutor(max_workers=2)` 进行 CPU 密集型计算。
*   **车间二 (OCR 协程)**：使用 `asyncio.Semaphore(4)` 控制并发请求数，高并发调用本地端口大模型服务，以压满 GPU。
*   **车间三 (组装器)**：事件驱动，单协程保序写盘，确保无多线程/多协程竞争文件写入。

## 3. 详细设计与数据契约

### 3.1 版面模型单例化 (`LayoutPredictor`)
在管线启动之初，在主线程加载版面检测模型，并在线程池内复用，消除多页 PDF 频繁加载模型的开销。
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

### 3.2 数据流契约
三级车间通过 `asyncio.Queue` 传递数据结构。

1.  **LayoutQueue 元素**:
    ```python
    {
        "page_idx": int,
        "img_path": str # 物理页面路径
    }
    ```

2.  **页面骨架 (Page Structure) — 送往组装器**:
    ```python
    {
        "page_idx": int,
        "page_size": [int, int], # [width, height]
        "blocks": [
            # image_block: 由车间一当场裁剪完成
            {
                "block_idx": int,
                "type": str,       # "figure" / "table" / "formula"
                "bbox": [int, int, int, int],
                "image_path": str,  # 裁剪保存的相对路径
            },
            # ocr_task: 待车间二 OCR 完毕后填充的占位块
            {
                "block_idx": int,
                "type": str,       # "text" / "table" / "formula" 等
                "bbox": [int, int, int, int],
                "content": None    # 待填充，初始为 None
            }
        ]
    }
    ```

3.  **OcrQueue 元素**:
    ```python
    {
        "page_idx": int,
        "block_idx": int,
        "label": str,       # 模型预测类别
        "image": PIL.Image  # crop 出来的待 OCR 子图对象或临时路径
    }
    ```

4.  **OCR 结果**:
    ```python
    {
        "page_idx": int,
        "block_idx": int,
        "content": str      # OCR 文本内容
    }
    ```

## 4. 组装与保序流式写盘机制 (`StreamAssembler`)
为防止多线程/多协程异步环境下结果返回顺序随机错乱，`StreamAssembler` 在内存中维护缓冲结构：
1.  **注册骨架**：车间一在分析完 `page_idx` 后，立即调用 `register_page` 录入骨架，计算本页待完成的 `pending_ocr` 任务数。
2.  **异步填充**：车间二完成 OCR 后，调用 `fill_ocr_content` 填充 `content`，`pending_ocr` 自减 1。
3.  **保序输出**：每次状态变化后，触发 `flush_ready_pages` 尝试输出。当 `current_writing_page` 页面的所有 `ocr_task` 均被填充（即 `pending_ocr == 0`），将其按顺序拼装成 Markdown 并**追加物理写入 `final_output.md`**，随后递增 `current_writing_page`，循环此操作。
4.  **最终归档**：
    *   所有页面处理完毕（`current_writing_page == total_pages`），将缓冲区的完整数据整理为中介 JSON 写入 `_middle.json`。
    *   启动 Pandoc 任务将 `final_output.md` 一切性转换导出为 Word。

## 5. 异常处理与容错
1.  **Layout 失败**：若某一页 Layout 解析抛错，生成一条占位信息，直接将该页标记为就绪，以防流水线死锁。
2.  **OCR 失败**：若 API 触发超时或异常，捕获异常并回填 `[OCR识别失败]` 提示，不阻断流水线。
3.  **缓存命中**：在管线入口处，若中介 JSON 存在且未指定 `--force`，仍走极速渲染逻辑，直接秒级输出 md 与 docx。
