# appocr_vllm_ui.py (GLM-OCR 适配版 - 使用 vLLM OpenAI 兼容 API)

import gradio as gr
import requests
import base64
import time
import logging
import os
from pathlib import Path
import shutil
import tempfile

# ✨ 导入前后处理工具 (GLM-OCR 不需要 Grounding 相关函数)
from concurrent.futures import ThreadPoolExecutor, as_completed
from glmocr.preprocessing import convert_pdf_to_images
from glmocr.postprocessing import merge_markdown_files, convert_file_with_pandoc
from glmocr.config import VLLM_API_URL, MODEL_NAME

# --- (全局配置) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [GLM_OCR_UI] - %(message)s')
MERGE_MODE_SINGLE_STREAM = "合并为单页流式文档 (默认)"
MERGE_MODE_PAGINATED = "保留分页 (每页PDF一页Word)"

# ✨ GLM-OCR 支持的 Prompt 模式
OCR_PROMPTS = {
    "文本识别": "Text Recognition:",
    "公式识别": "Formula Recognition:",
    "表格识别": "Table Recognition:",
}


# --- (辅助函数) ---
def parse_page_selection(page_str):
    """
    解析页码字符串，例如 "1, 3-5, 10" -> [0, 2, 3, 4, 9] (转为0-based索引)
    """
    if not page_str or not page_str.strip():
        return None
    
    selected = set()
    parts = page_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                for i in range(start, end + 1):
                    selected.add(i - 1)
            except ValueError:
                continue
        else:
            try:
                page = int(part)
                selected.add(page - 1)
            except ValueError:
                continue
    
    return sorted([p for p in list(selected) if p >= 0])


def call_glmocr_api(img_base64: str, prompt: str, max_tokens: int = 1536) -> str:
    """
    调用 vLLM OpenAI 兼容 API 进行 GLM-OCR 推理。
    
    Args:
        img_base64: Base64 编码的图片
        prompt: GLM-OCR 的 Prompt (如 "Text Recognition:")
        max_tokens: 最大输出 token 数 (由于图片输入占了约 6k，输出必须限制在 2k 以内)
    
    Returns:
        模型输出的文本
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    response = requests.post(VLLM_API_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    
    # OpenAI 格式的响应
    return data["choices"][0]["message"]["content"]


# --- (核心处理函数) ---
def ocr_pipeline_vllm(input_files, input_path_textbox, page_range_str, output_root_dir, ocr_mode, convert_format, pagination_mode):
    log_update = "任务初始化...\n"
    yield "任务初始化...", log_update, gr.update(value=None, visible=False)
    
    overall_start_time = time.time()
    try:
        # --- 源路径处理 ---
        source_path = None
        if input_files:
            source_path = Path(input_files[0].name)
            if len(input_files) > 1 or not source_path.is_file():
                source_path = source_path.parent
        elif input_path_textbox:
            source_path = Path(input_path_textbox.strip())
        else:
            raise ValueError("错误：请提供输入源。")
            
        if not source_path.exists():
            raise FileNotFoundError(f"错误：输入路径不存在: {source_path}")
            
        project_name = source_path.stem if source_path.is_file() else source_path.name
        log_update += f"项目名称: {project_name}\n"
        
        if output_root_dir:
            final_output_path = Path(output_root_dir.strip()) / project_name
        else:
            final_output_path = source_path.parent / project_name if source_path.is_file() else source_path / "OCR_Output"
            
        final_output_path.mkdir(parents=True, exist_ok=True)
        log_update += f"结果将保存至: {final_output_path}\n"
        yield "准备文件中...", log_update, None
        
        with tempfile.TemporaryDirectory() as temp_dir_str:
            # --- 预处理 ---
            temp_dir = Path(temp_dir_str)
            image_folder_path = temp_dir / "images_to_process"
            image_folder_path.mkdir()
            is_single_image = False
            
            target_pages = parse_page_selection(page_range_str)
            if target_pages:
                log_update += f"指定处理页码 (索引): {target_pages}\n"

            if source_path.is_dir():
                log_update += f"处理文件夹 (页码选择对文件夹模式无效)...\n"
                shutil.copytree(source_path, image_folder_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('OCR_Output*'))
            elif source_path.is_file():
                if source_path.suffix.lower() == '.pdf':
                    log_update += "检测到PDF文件...\n"
                    yield "PDF转换中...", log_update, None
                    convert_pdf_to_images(source_path, image_folder_path, target_pages=target_pages)
                else:
                    is_single_image = True
                    log_update += f"处理单张图片...\n"
                    shutil.copy(source_path, image_folder_path / source_path.name)
            
            image_paths = sorted([p for p in image_folder_path.rglob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']])
            if not image_paths:
                raise FileNotFoundError("未找到有效图片（可能是页码超出范围或PDF为空）。")
            
            total_images = len(image_paths)
            md_output_path = temp_dir / "markdown_results"
            md_output_path.mkdir()
            
            # ✨ 获取 GLM-OCR Prompt
            prompt = OCR_PROMPTS.get(ocr_mode, "Text Recognition:")
            log_update += f"预处理完成，发现 {total_images} 张图片。Prompt: {prompt}\n"
            log_update += f"启动并发推理...\n"
            yield f"并发推理中...", log_update, None
            
            # --- ✨ 定义单张图片处理函数 (Worker) ✨ ---
            def process_single_image(img_path, index):
                """
                GLM-OCR 版：读图 -> 发 OpenAI API 请求 -> 存结果
                不需要 Grounding 后处理
                """
                try:
                    # 1. 编码图片并在需要时进行等比例缩放限制
                    from PIL import Image
                    import io
                    max_dim = 1536 # 这个尺寸对应的大约是 4000-5000 Tokens 内，绝对安全
                    with Image.open(img_path) as img:
                        if img.width > max_dim or img.height > max_dim:
                            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        
                        buffered = io.BytesIO()
                        img.convert("RGB").save(buffered, format="JPEG", quality=85)
                        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    # 2. 错峰
                    if index % 2 != 0: 
                        time.sleep(0.2) 
                    
                    # 3. 调用 GLM-OCR API (OpenAI 兼容格式)
                    result_text = call_glmocr_api(img_base64, prompt)
                    
                    # 4. 保存 Markdown
                    temp_md_path = md_output_path / f"{img_path.stem}.md"
                    temp_md_path.write_text(result_text, encoding="utf-8")
                    
                    # 5. 复制到最终目录
                    shutil.copy(temp_md_path, final_output_path / temp_md_path.name)
                    return {"success": True, "file": img_path.name}
                        
                except Exception as e:
                    return {"success": False, "file": img_path.name, "error": str(e)}

            # --- 🚀 启动线程池 ---
            # GLM-OCR 只有 0.9B，显存占用小，可以多并发
            completed_count = 0
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_file = {
                    executor.submit(process_single_image, img_path, i): img_path 
                    for i, img_path in enumerate(image_paths)
                }
                
                for future in as_completed(future_to_file):
                    completed_count += 1
                    result = future.result()
                    
                    if result["success"]:
                        log_msg = f"  ✅ [{completed_count}/{total_images}] 完成: {result['file']}\n"
                    else:
                        log_msg = f"  ❌ [{completed_count}/{total_images}] 失败: {result['file']} - {result.get('error')}\n"
                    
                    log_update += log_msg
                    yield f"进度: {completed_count}/{total_images}", log_update, None

            log_update += "所有并发任务完成，开始最终合并...\n"
            yield "最终合并中...", log_update, None
            
            final_file_to_download = None
            # --- 后处理 ---
            if is_single_image:
                final_file_to_download = next(final_output_path.glob("*.md"), None)
                
                if final_file_to_download and "仅生成 Markdown" not in convert_format:
                    convert_option = "docx" if "Word" in convert_format else "html"
                    final_file_to_download = convert_file_with_pandoc(
                        final_file_to_download, 
                        convert_option, 
                        use_lua=False
                    )
            else:
                merged_md_path = final_output_path / f"{project_name}_merged.md"
                
                is_paginated = (pagination_mode == MERGE_MODE_PAGINATED)
                internal_mode = "paginated" if is_paginated else "stream"
                
                merge_markdown_files(final_output_path, merged_md_path, mode=internal_mode)
                
                if "仅生成 Markdown" in convert_format:
                    final_file_to_download = merged_md_path
                else:
                    convert_option = "docx" if "Word" in convert_format else "html"
                    should_use_lua = is_paginated and (convert_option == "docx")
                    
                    final_file_to_download = convert_file_with_pandoc(
                        merged_md_path, 
                        convert_option,
                        use_lua=should_use_lua
                    )

            # --- 结果返回 ---
            overall_end_time = time.time()
            total_time = overall_end_time - overall_start_time
            preview_update = (f"🎉 任务成功！总耗时: {total_time:.2f} 秒\n结果保存在: '{final_output_path}'")
            log_update += f"🎉 任务成功！总耗时: {total_time:.2f} 秒\n"
            
            file_update = gr.update(value=None, visible=False)
            if final_file_to_download and final_file_to_download.exists():
                temp_safe_path = shutil.copy(final_file_to_download, temp_dir)
                file_update = gr.update(value=temp_safe_path, visible=True)
            
            yield preview_update, log_update, file_update
            
    except Exception as e:
        logging.error(f"处理流程发生严重错误: {e}", exc_info=True)
        error_message = f"❌ 错误: {str(e)}"
        yield error_message, log_update + f"\n{error_message}", gr.update(value=None, visible=False)

# --- (Gradio 界面构建) ---
with gr.Blocks() as iface:
    gr.Markdown("# 🚀 GLM-OCR 文档识别系统 (vLLM Backend)")
    gr.Markdown("基于 GLM-OCR 0.9B SOTA 模型 | OmniDocBench V1.5 #1 | 前后端分离架构")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. 输入源")
            uploaded_files = gr.File(label="上传文件 (PDF/图片) 或文件夹", file_count="multiple")
            path_input_textbox = gr.Textbox(label="或 输入本地文件/文件夹的完整路径")
            
            page_range_textbox = gr.Textbox(
                label="指定 PDF 页码 (可选)", 
                placeholder="例如: 1, 3-5, 10 (留空则处理所有页)",
                info="仅对 PDF 有效。支持逗号分隔和连字符范围。"
            )
            
            gr.Markdown("### 2. 输出选项")
            output_dir_textbox = gr.Textbox(
                label="指定输出根目录 (留空则在源文件旁生成)",
                info="例如: E:\\project\\GLM-OCR\\ocr_results",
                value=""
            )
            # ✨ GLM-OCR 模式选择：使用固定 Prompt
            ocr_mode_radio = gr.Radio(
                ["文本识别", "公式识别", "表格识别"], 
                label="选择识别模式", 
                value="文本识别",
                info="文本识别=通用文档, 公式识别=数学公式, 表格识别=表格提取"
            )
            convert_format_radio = gr.Radio(
                ["仅生成 Markdown", "转换为 Word (.docx)", "转换为 HTML (.html)"], 
                label="选择最终输出格式", 
                value="仅生成 Markdown"
            )
            pagination_radio = gr.Radio(
                [MERGE_MODE_SINGLE_STREAM, MERGE_MODE_PAGINATED],
                label="多页文档处理模式",
                value=MERGE_MODE_SINGLE_STREAM,
                info="决定最终Word/HTML文档是连续的还是严格分页的。"
            )
            submit_btn = gr.Button("开始识别", variant="primary", scale=2)

        with gr.Column(scale=3):
            gr.Markdown("### 3. 结果与进度")
            result_preview_textbox = gr.Textbox(label="任务状态与摘要", lines=10, interactive=False)
            log_textbox = gr.Textbox(label="实时日志", lines=12, interactive=False, max_lines=20)
            result_file_output = gr.File(label="下载最终结果文件", visible=True)

    submit_btn.click(
        fn=ocr_pipeline_vllm,
        inputs=[
            uploaded_files, 
            path_input_textbox,
            page_range_textbox,
            output_dir_textbox,
            ocr_mode_radio, 
            convert_format_radio,
            pagination_radio
        ],
        outputs=[result_preview_textbox, log_textbox, result_file_output]
    )

if __name__ == "__main__":
    iface.launch()
