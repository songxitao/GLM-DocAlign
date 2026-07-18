import os
import torch
import asyncio
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from glmocr.pipeline.deskew import detect_skew_angle, rotate_image
from glmocr.pipeline.xycut import sort_boxes_by_xy_cut
from glmocr.pipeline.masked_crop import crop_and_mask
from glmocr.pipeline.async_ocr import run_async_ocr, ocr_single_image

from glmocr.config import LOCAL_LAYOUT_MODEL
_global_predictor = None


class LayoutPredictor:
    def __init__(self, model_dir: str):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"\n[FAIL] 无法找到 Layout 模型目录: {model_dir}\n"
                f"本项目已将业务代码与大模型权重解耦。请执行以下步骤以物理关联模型：\n"
                f"1. 从 ModelScope(魔搭社区) 下载 PP-DocLayout-V3 模型权重。\n"
                f"2. 在项目根目录下创建 model/ 目录，并将解压后的模型放置于 model/PP-DocLayoutV3safetensor/\n"
                f"   (或者配置系统环境变量 LOCAL_LAYOUT_MODEL 指向您的自定义模型物理路径)\n"
                f"🔗 ModelScope 权重下载链接: https://modelscope.cn/models/songxitao/PP-DocLayoutV3safetensor\n"
            )
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir).to("cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)

    def predict(self, corrected_image):
        inputs = self.image_processor(images=corrected_image, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.image_processor.post_process_object_detection(
            outputs, target_sizes=[corrected_image.size[::-1]]
        )


def run_pipeline_flow(
    image_path: str,
    output_dir: str,
    page_idx: int = 0,
    table_as_image: bool = True,
    formula_as_image: bool = False,
    keep_header_footer: bool = False,
    predictor: LayoutPredictor = None
) -> tuple[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    images_subdir = os.path.join(output_dir, "images")
    os.makedirs(images_subdir, exist_ok=True)
    
    # 1. 加载图片并进行纠偏
    raw_image = Image.open(image_path).convert("RGB")
    angle = detect_skew_angle(raw_image)
    corrected_image = rotate_image(raw_image, -angle)
    
    # 2. 版面检测模型推理
    if predictor is None:
        global _global_predictor
        if _global_predictor is None:
            _global_predictor = LayoutPredictor(LOCAL_LAYOUT_MODEL)
        predictor = _global_predictor

    results = predictor.predict(corrected_image)
    
    boxes = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < 0.4:
                continue
            label = predictor.model.config.id2label.get(label_id.item(), f"Label_{label_id.item()}")
            box_coords = [int(i) for i in box.tolist()]
            boxes.append({"coords": box_coords, "label": label})
            
    if not boxes:
        return "⚠️ 未检测到任何版面框。", {"page_idx": page_idx, "page_size": list(corrected_image.size), "blocks": []}
        
    # 绘制画框诊断图并保存（以当前页面文件名命名，防止覆盖）
    from PIL import ImageDraw
    from pathlib import Path
    page_stem = Path(image_path).stem
    
    diag_img = corrected_image.copy()
    diag_draw = ImageDraw.Draw(diag_img)
    colors = {
        "text": "green",
        "table": "blue",
        "formula": "red",
        "figure": "purple",
        "image": "purple",
        "chart": "purple"
    }
    for box_item in boxes:
        coords = box_item["coords"]
        lbl = box_item["label"]
        color = colors.get(lbl.lower(), "yellow")
        diag_draw.rectangle(coords, outline=color, width=3)
        # 简单在框左上角标注类别名称
        diag_draw.text((coords[0], max(0, coords[1] - 12)), lbl, fill=color)
        
    # 归档诊断图：创建专门的 diagnostics 目录保存画框诊断图
    diagnostics_subdir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_subdir, exist_ok=True)
    
    diag_filename = f"{page_stem}_diagnostic.png"
    diag_path = os.path.join(diagnostics_subdir, diag_filename)
    diag_img.save(diag_path)
        
    # 3. XY-Cut 重排序
    sorted_indices = sort_boxes_by_xy_cut(boxes)
    
    # 4. 裁剪并生成处理列表
    final_elements = [] 
    fig_counter = 1
    table_counter = 1
    formula_counter = 1
    
    for idx in sorted_indices:
        element = boxes[idx]
        label = element["label"]
        lbl_lower = label.lower()
        
        # 页眉页脚过滤控制
        if not keep_header_footer:
            # 智能页眉过滤：如果是单栏表格账单（有 table），页眉通常是重要的“要货单位”、“发票抬头”等，不能丢弃。
            # 我们只在普通学术文档（无 table）中才过滤 header，在账本中予以保留。
            has_table = any(b["label"].lower() == "table" for b in boxes)
            if lbl_lower == "footer" or (lbl_lower == "header" and not has_table):
                continue
            
        is_crop = False
        clean_tag = ""
        markdown_label = ""
        counter = 0
        
        if lbl_lower in ["figure", "image", "chart"]:
            is_crop = True
            clean_tag = "fig"
            markdown_label = "figure"
            counter = fig_counter
            fig_counter += 1
        elif lbl_lower == "table" and table_as_image:
            is_crop = True
            clean_tag = "table"
            markdown_label = "table"
            counter = table_counter
            table_counter += 1
        elif lbl_lower == "formula" and formula_as_image:
            is_crop = True
            clean_tag = "formula"
            markdown_label = "formula"
            counter = formula_counter
            formula_counter += 1
            
        if is_crop:
            # 物理裁剪保存
            filename = f"{page_stem}_{clean_tag}_{counter}.png"
            fig_path = os.path.join(images_subdir, filename)
            cropped_fig = corrected_image.crop(element["coords"])
            cropped_fig.save(fig_path)
            
            # 自适应图片尺寸控制：计算检测框在页面中的相对宽度占比，自适应注入 Pandoc 尺寸属性
            width_str = ""
            if clean_tag in ["fig", "table"]:
                page_width = corrected_image.size[0]
                box_coords = element["coords"]
                box_width = box_coords[2] - box_coords[0]
                ratio = float(box_width) / page_width
                
                if ratio >= 0.65:
                    width_str = "{width=100%}"
                elif ratio >= 0.35:
                    width_str = "{width=70%}"
                else:
                    width_str = "{width=45%}"
            
            markdown_content = f"\n\n![{markdown_label}](images/{filename}){width_str}\n\n"
            final_elements.append({
                "type": "image_block",
                "label": label,
                "bbox": element["coords"],
                "image_path": f"images/{filename}",
                "markdown_content": markdown_content
            })
        else:
            # 文本、公式、表格、摘要等：裁剪去噪并送入 OCR 推理
            cropped_sub = crop_and_mask(corrected_image, boxes, idx)
            final_elements.append({
                "type": "ocr_task",
                "label": label,
                "bbox": element["coords"],
                "image": cropped_sub
            })
            
    # 5. 提取需要 OCR 的子图片并进行异步并发识别
    ocr_images_info = [
        {"path": el["image"], "label": el["label"]}
        for el in final_elements if el["type"] == "ocr_task"
    ]
    
    ocr_texts = []
    if ocr_images_info:
        ocr_texts = asyncio.run(run_async_ocr(ocr_images_info, concurrency=4))
        
    # 6. 将结果拼装为 Markdown 并构造 Middle JSON 数据
    page_middle_data = {
        "page_idx": page_idx,
        "page_size": list(corrected_image.size),  # [width, height]
        "blocks": []
    }
    
    ocr_idx = 0
    markdown_lines = []
    for el in final_elements:
        if el["type"] == "image_block":
            markdown_lines.append(el["markdown_content"])
            block = {
                "type": el["label"].lower(),
                "bbox": el["bbox"],
                "image_path": el["image_path"]
            }
            page_middle_data["blocks"].append(block)
        elif el["type"] == "ocr_task":
            # 兼容 mock 出来的长度不一致，或者出现异常的情况
            if ocr_idx < len(ocr_texts):
                txt = ocr_texts[ocr_idx].strip()
                lbl = el["label"].lower()
                
                # 强力防御：清洗 VLM 幻觉或带入的反引号，防止破坏全局 Markdown 代码块的闭合性
                txt = txt.replace("```", "")
                
                if not txt:
                    # 强力降噪：如果识别出的文本内容为空，直接丢弃该块，杜绝产生空 Markdown 代码块与格式垃圾
                    ocr_idx += 1
                    continue
                
                # 根据版面模型提供的语义标签，自动加注 Markdown 格式符号
                if lbl == "doc_title":
                    markdown_lines.append(f"\n\n# {txt}\n\n")
                elif lbl == "paragraph_title":
                    markdown_lines.append(f"\n\n## {txt}\n\n")
                elif lbl == "table":
                    # 确保 Markdown 表格前后有充足的空行，以便 Pandoc 完美解析为 Word 原生可编辑表格
                    markdown_lines.append(f"\n\n{txt}\n\n")
                elif lbl == "abstract":
                    # 摘要段落：映射为 Markdown 的 Blockquote（引用块），激活 Word 的引文缩进样式
                    markdown_lines.append(f"\n\n> **Abstract** — {txt}\n\n")
                elif lbl == "algorithm":
                    # 算法段落：用代码块包装，激活等宽和底框样式
                    markdown_lines.append(f"\n\n```\n{txt}\n```\n\n")
                elif lbl == "figure_title":
                    # 图表标题：斜体处理，并紧贴下方或上方，对应 Caption 样式
                    markdown_lines.append(f"\n\n*{txt}*\n\n")
                elif lbl == "reference_content":
                    # 参考文献内容：直接输出
                    markdown_lines.append(f"\n\n{txt}\n\n")
                else:
                    # 兜底普通文本：确保前后换行保护，防止折行变为软回车
                    markdown_lines.append(f"\n\n{txt}\n\n")
                
                block = {
                    "type": el["label"].lower(),
                    "bbox": el["bbox"],
                    "content": txt
                }
                page_middle_data["blocks"].append(block)
            else:
                fail_msg = f"[OCR识别失败：索引溢出，标签为: {el['label']}]"
                markdown_lines.append(f"\n\n{fail_msg}\n\n")
                block = {
                    "type": el["label"].lower(),
                    "bbox": el["bbox"],
                    "content": fail_msg
                }
                page_middle_data["blocks"].append(block)
            ocr_idx += 1
            
    full_markdown = "\n\n".join(markdown_lines)
    return full_markdown, page_middle_data


from pathlib import Path
from glmocr.postprocessing import smart_reflow_markdown

class StreamAssembler:
    def __init__(self, output_dir, stem, total_pages):
        self.output_dir = Path(output_dir)
        self.stem = stem
        self.total_pages = total_pages
        self.page_buffers = {}
        self.final_pdf_info = [None] * total_pages
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
                "ready": ocr_tasks_count == 0,
                "filled_blocks": set()
            }
            await self.flush_ready_pages()

    async def fill_ocr_content(self, page_idx, block_idx, content):
        async with self.lock:
            if page_idx not in self.page_buffers:
                return
            page = self.page_buffers[page_idx]
            if block_idx in page["filled_blocks"]:
                return
            
            matched = False
            for block in page["structure"]["blocks"]:
                if block.get("block_idx") == block_idx:
                    block["content"] = content.replace("```", "")
                    matched = True
                    break
            
            if matched:
                page["filled_blocks"].add(block_idx)
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
            self.final_pdf_info[self.current_writing_page] = page["structure"]
            del self.page_buffers[self.current_writing_page]
            
            self.current_writing_page += 1
            if self.current_writing_page == self.total_pages:
                self.finished_event.set()

    def _write_file_sync(self, content):
        with open(self.output_md_path, "a", encoding="utf-8") as f:
            f.write(content)

    async def write_page_to_disk(self, page_idx, structure):
        markdown_lines = []
        for block in structure["blocks"]:
            if "content" in block:
                txt = block["content"].strip()
                lbl = block.get("label", "paragraph").lower()
                if not txt:
                    continue
                reflowed = smart_reflow_markdown(txt)
                
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
                lbl = block.get("label", "figure").lower()
                width_str = block.get("width_str", "")
                markdown_lines.append(f"\n\n![{lbl}]({block['image_path']}){width_str}\n\n")
        
        page_md = "\n\n".join(markdown_lines)
        if page_idx > 0:
            page_md = f"\n\n\\newpage\n\n{page_md}"
        await asyncio.to_thread(self._write_file_sync, page_md)


def process_single_page_layout(
    image_path: str,
    page_idx: int,
    predictor: LayoutPredictor,
    output_dir: str,
    keep_header_footer: bool = False,
    table_as_image: bool = True,
    formula_as_image: bool = False
) -> tuple[dict, list]:
    raw_image = Image.open(image_path).convert("RGB")
    angle = detect_skew_angle(raw_image)
    corrected_image = rotate_image(raw_image, -angle)
    
    results = predictor.predict(corrected_image)
    
    boxes = []
    for result in results:
        if not all(k in result for k in ["scores", "labels", "boxes"]):
            continue
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < 0.4:
                continue
            label = predictor.model.config.id2label.get(label_id.item(), f"Label_{label_id.item()}")
            box_coords = [int(i) for i in box.tolist()]
            boxes.append({"coords": box_coords, "label": label})
            
    page_stem = Path(image_path).name.rsplit(".", 1)[0]
    
    if not boxes:
        return {
            "page_idx": page_idx,
            "page_size": list(corrected_image.size),
            "blocks": []
        }, []
        
    # 绘制画框诊断图并保存
    from PIL import ImageDraw
    diag_img = corrected_image.copy()
    diag_draw = ImageDraw.Draw(diag_img)
    colors = {
        "text": "green",
        "table": "blue",
        "formula": "red",
        "figure": "purple",
        "image": "purple",
        "chart": "purple"
    }
    for box_item in boxes:
        coords = box_item["coords"]
        lbl = box_item["label"]
        color = colors.get(lbl.lower(), "yellow")
        diag_draw.rectangle(coords, outline=color, width=3)
        diag_draw.text((coords[0], max(0, coords[1] - 12)), lbl, fill=color)
        
    diagnostics_subdir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_subdir, exist_ok=True)
    diag_filename = f"{page_stem}_diagnostic.png"
    diag_path = os.path.join(diagnostics_subdir, diag_filename)
    diag_img.save(diag_path)
    
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
        
        if not keep_header_footer:
            has_table = any(b["label"].lower() == "table" for b in boxes)
            if lbl_lower == "footer" or (lbl_lower == "header" and not has_table):
                continue
                
        is_crop = False
        clean_tag = ""
        markdown_label = ""
        counter = 0
        
        if lbl_lower in ["figure", "image", "chart"]:
            is_crop = True
            clean_tag = "fig"
            markdown_label = "figure"
            counter = fig_counter
            fig_counter += 1
        elif lbl_lower == "table" and table_as_image:
            is_crop = True
            clean_tag = "table"
            markdown_label = "table"
            counter = table_counter
            table_counter += 1
        elif lbl_lower == "formula" and formula_as_image:
            is_crop = True
            clean_tag = "formula"
            markdown_label = "formula"
            counter = formula_counter
            formula_counter += 1
            
        if is_crop:
            filename = f"{page_stem}_{clean_tag}_{counter}.png"
            images_subdir = os.path.join(output_dir, "images")
            os.makedirs(images_subdir, exist_ok=True)
            fig_path = os.path.join(images_subdir, filename)
            cropped_fig = corrected_image.crop(element["coords"])
            cropped_fig.save(fig_path)
            
            width_str = ""
            if clean_tag in ["fig", "table"]:
                page_width = corrected_image.size[0]
                box_coords = element["coords"]
                box_width = box_coords[2] - box_coords[0]
                ratio = float(box_width) / page_width
                
                if ratio >= 0.65:
                    width_str = "{width=100%}"
                elif ratio >= 0.35:
                    width_str = "{width=70%}"
                else:
                    width_str = "{width=45%}"
                    
            page_structure["blocks"].append({
                "block_idx": block_idx,
                "type": "image_block",
                "label": label,
                "bbox": element["coords"],
                "image_path": f"images/{filename}",
                "width_str": width_str
            })
        else:
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


async def run_pipeline_flow_async(
    img_files: list,
    output_dir: str,
    stem: str,
    table_as_image: bool = True,
    formula_as_image: bool = False,
    keep_header_footer: bool = False,
    max_layout_workers: int = 2,
    ocr_concurrency: int = 4
) -> list:
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = LayoutPredictor(LOCAL_LAYOUT_MODEL)
        
    total_pages = len(img_files)
    assembler = StreamAssembler(output_dir, stem, total_pages)
    
    layout_queue = asyncio.Queue()
    ocr_queue = asyncio.Queue()
    
    for idx, img_path in enumerate(img_files):
        await layout_queue.put({"page_idx": idx, "img_path": str(img_path)})
        
    loop = asyncio.get_running_loop()
    from concurrent.futures import ThreadPoolExecutor
    thread_pool = ThreadPoolExecutor(max_workers=max_layout_workers)
    
    async def layout_worker():
        while True:
            page_idx = -1
            got_task = False
            try:
                task = await layout_queue.get()
                got_task = True
                page_idx = task["page_idx"]
                img_path = task["img_path"]
                
                page_structure, ocr_tasks = await loop.run_in_executor(
                    thread_pool,
                    process_single_page_layout,
                    img_path, page_idx, _global_predictor, output_dir, keep_header_footer, table_as_image, formula_as_image
                )
                
                await assembler.register_page(page_idx, page_structure)
                for ocr_task in ocr_tasks:
                    await ocr_queue.put(ocr_task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[FAIL] Layout 异常 (Page {page_idx}): {e}")
                if page_idx != -1:
                    err_structure = {
                        "page_idx": page_idx,
                        "page_size": [100, 100],
                        "blocks": [{"block_idx": 0, "type": "ocr_task", "label": "paragraph", "bbox": [0,0,10,10], "content": f"[Layout分析异常: {e}]"}]
                    }
                    await assembler.register_page(page_idx, err_structure)
            finally:
                if got_task:
                    layout_queue.task_done()
                
    sem = asyncio.Semaphore(ocr_concurrency)
    import aiohttp
    async def ocr_worker():
        async with aiohttp.ClientSession() as session:
            while True:
                got_task = False
                try:
                    task = await ocr_queue.get()
                    got_task = True
                    page_idx = task["page_idx"]
                    block_idx = task["block_idx"]
                    label = task["label"]
                    img_obj = task["image"]
                    
                    content = await ocr_single_image(session, img_obj, label, sem)
                    await assembler.fill_ocr_content(page_idx, block_idx, content)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[FAIL] OCR 异常 (Page {page_idx}, Block {block_idx}): {e}")
                    await assembler.fill_ocr_content(page_idx, block_idx, f"[OCR识别失败: {e}]")
                finally:
                    if got_task:
                        ocr_queue.task_done()
                    
    layout_tasks = [asyncio.create_task(layout_worker()) for _ in range(max_layout_workers)]
    ocr_tasks = [asyncio.create_task(ocr_worker()) for _ in range(ocr_concurrency)]
    
    await layout_queue.join()
    await ocr_queue.join()
    await assembler.finished_event.wait()
    
    for w in layout_tasks:
        w.cancel()
    for w in ocr_tasks:
        w.cancel()
    thread_pool.shutdown()
    
    return assembler.final_pdf_info



