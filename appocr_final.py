# appocr_final.py (The Definitive Edition, calling the correct backend)
import gradio as gr
import requests
import base64
import time
import logging
import os
from pathlib import Path
import shutil
import tempfile
from tqdm import tqdm

from preprocessing import convert_pdf_to_images
from postprocessing import merge_markdown_files, convert_file_with_pandoc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [OCR_UI_FINAL] - %(message)s')
VLLM_SERVER_URL = "http://127.0.0.1:8000/ocr"

def ocr_pipeline(input_files, input_path_textbox, ocr_mode, convert_format):
    start_time = time.time()
    log_update = ""
    preview_update = "任务初始化..."
    yield preview_update, log_update, gr.update(value=None, visible=False)

    try:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            # 文件预处理逻辑 (与之前版本相同)
            temp_dir = Path(temp_dir_str)
            image_folder_path = temp_dir / "images_to_process"
            md_output_path = temp_dir / "markdown_results"
            image_folder_path.mkdir(); md_output_path.mkdir()
            is_single_image = False
            project_name = f"ocr_project_{int(time.time())}"
            source_path = None
            if input_files:
                source_path = Path(input_files[0].name)
                if len(input_files) > 1 or source_path.is_dir(): source_path = source_path.parent
            elif input_path_textbox: source_path = Path(input_path_textbox.strip())
            else: raise ValueError("错误：请提供输入。")
            if not source_path.exists(): raise FileNotFoundError(f"错误：输入路径不存在: {source_path}")
            if source_path.is_dir():
                project_name = source_path.name; log_update += f"处理文件夹: {project_name}\n"
                shutil.copytree(source_path, image_folder_path, dirs_exist_ok=True)
            elif source_path.is_file():
                project_name = source_path.stem
                if source_path.suffix.lower() == '.pdf':
                    log_update += f"检测到PDF文件，开始转换...\n"; yield preview_update, log_update, None
                    convert_pdf_to_images(source_path, image_folder_path)
                else:
                    is_single_image = True; log_update += f"处理单张图片: {source_path.name}\n"
                    shutil.copy(source_path, image_folder_path)
            image_paths = sorted([p for p in image_folder_path.rglob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']])
            if not image_paths: raise FileNotFoundError("未找到任何有效图片。")
            
            # 核心OCR处理
            total_images = len(image_paths)
            log_update += f"预处理完成，发现 {total_images} 张图片。开始提交给vLLM服务...\n"; yield preview_update, log_update, None

            for i, img_path in enumerate(tqdm(image_paths, desc="OCR Progress")):
                log_update += f"  - 正在处理: {img_path.name} ({i+1}/{total_images})\n"
                preview_update = f"进度: {i+1}/{total_images}\n正在识别: {img_path.name}"
                yield preview_update, log_update, None

                with open(img_path, "rb") as image_file:
                    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
                # ✨ 关键：API 调用现在发送 mode 参数 ✨
                payload = {"image_base64": img_base64, "mode": ocr_mode}
                response = requests.post(VLLM_SERVER_URL, json=payload, timeout=180)
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'success':
                    with open(md_output_path / f"{img_path.stem}.md", "w", encoding="utf-8") as f:
                        f.write(data['result'])
                else:
                    log_update += f"  - 警告: {img_path.name} 处理失败: {data.get('message')}\n"

            # 后处理逻辑 (与之前版本相同)
            output_dir = Path(os.path.expanduser("~")) / "Downloads" / f"OCR_Output_{project_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            final_file_to_download = None
            if is_single_image:
                md_file = list(md_output_path.glob("*.md"))[0]; final_file_to_download = shutil.copy(md_file, output_dir)
            else:
                merged_md_path = output_dir / f"{project_name}_merged.md"; merge_markdown_files(md_output_path, merged_md_path); final_file_to_download = merged_md_path
            if "Word" in convert_format or "HTML" in convert_format:
                convert_option = "docx" if "Word" in convert_format else "html"; convert_file_with_pandoc(final_file_to_download, convert_option); final_file_to_download = final_file_to_download.with_suffix(f".{convert_option}")
            end_time = time.time(); total_time = end_time - start_time
            preview_update = (f"🎉 任务成功！总耗时: {total_time:.2f} 秒, 平均速度: {total_images/total_time:.2f} 页/秒\n"
                            f"结果保存在: '下载' 文件夹 -> {output_dir.name}")
            yield preview_update, log_update, gr.update(value=str(final_file_to_download), visible=True)

    except Exception as e:
        logging.error(f"处理流程发生严重错误: {e}", exc_info=True)
        preview_update = f"❌ 处理过程中发生错误: {str(e)}"
        yield preview_update, log_update, gr.update(value=None, visible=False)

# Gradio 界面构建 (与之前版本相同)
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🚀 Ultimate DeepSeek-OCR (vLLM Official Method)")
    # ... (此处省略与之前版本相同的Gradio布局代码，确保 ocr_mode_radio 是 interactive=True)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1. 输入源"); uploaded_files = gr.File(label="...", file_count="multiple"); path_input_textbox = gr.Textbox(label="...")
            gr.Markdown("### 2. 输出选项")
            ocr_mode_radio = gr.Radio(["tiny", "small", "base", "large", "gundam"], label="选择模式", value="base", interactive=True, info="不同的模式会影响后端 logits processor 的参数。")
            convert_format_radio = gr.Radio(["...", "...", "..."], label="选择最终输出格式", value="...")
            submit_btn = gr.Button("开始识别", variant="primary", scale=2)
        with gr.Column(scale=3):
            gr.Markdown("### 3. 结果与进度"); result_preview_textbox = gr.Textbox(label="...", lines=10); log_textbox = gr.Textbox(label="...", lines=12); result_file_output = gr.File(label="...", visible=False)
    submit_btn.click(fn=ocr_pipeline, inputs=[uploaded_files, path_input_textbox, ocr_mode_radio, convert_format_radio], outputs=[result_preview_textbox, log_textbox, result_file_output])

if __name__ == "__main__":
    iface.launch()