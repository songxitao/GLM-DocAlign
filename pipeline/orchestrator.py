import os
import torch
import asyncio
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from pipeline.deskew import detect_skew_angle, rotate_image
from pipeline.xycut import sort_boxes_by_xy_cut
from pipeline.masked_crop import crop_and_mask
from pipeline.async_ocr import run_async_ocr

LOCAL_LAYOUT_MODEL = r"E:\project\GLM-OCR\model\PP-DocLayoutV3safetensor"

def run_pipeline_flow(
    image_path: str,
    output_dir: str,
    page_idx: int = 0,
    table_as_image: bool = True,
    formula_as_image: bool = False,
    keep_header_footer: bool = False
) -> tuple[str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    images_subdir = os.path.join(output_dir, "images")
    os.makedirs(images_subdir, exist_ok=True)
    
    # 1. 加载图片并进行纠偏
    raw_image = Image.open(image_path).convert("RGB")
    angle = detect_skew_angle(raw_image)
    corrected_image = rotate_image(raw_image, -angle)
    
    # 2. 本地加载版面检测模型并推理
    model = AutoModelForObjectDetection.from_pretrained(LOCAL_LAYOUT_MODEL).to("cpu")
    image_processor = AutoImageProcessor.from_pretrained(LOCAL_LAYOUT_MODEL)
    
    inputs = image_processor(images=corrected_image, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = image_processor.post_process_object_detection(outputs, target_sizes=[corrected_image.size[::-1]])
    
    boxes = []
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            if score.item() < 0.4:
                continue
            label = model.config.id2label.get(label_id.item(), f"Label_{label_id.item()}")
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
