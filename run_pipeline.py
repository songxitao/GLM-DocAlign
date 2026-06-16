import os
import sys
import asyncio
from pathlib import Path
from preprocessing import convert_pdf_to_images
from postprocessing import smart_reflow_markdown, convert_file_with_pandoc
from pipeline.orchestrator import run_pipeline_flow

def main():
    import json
    
    # 提取并移除 --force 参数
    force = "--force" in sys.argv
    if force:
        sys.argv.remove("--force")

    if len(sys.argv) < 2:
        print("📢 用法: python run_pipeline.py <PDF路径或图片路径> [输出目录] [--force]")
        sys.exit(1)
        
    input_path = Path(sys.argv[1])
    stem = input_path.stem
    
    # 智能命名：使用输入文件名（去后缀的 stem）命名输出目录，实现更好的文件归档
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2]) / stem
    else:
        output_dir = Path("ocr_output") / stem
    
    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        sys.exit(1)
        
    middle_json_path = output_dir / f"{stem}_middle.json"
    
    # 3. 改造缓存命中后的本地免 API 渲染逻辑
    if middle_json_path.exists() and not force:
        print("📢 检测到本地缓存，正在直接渲染 Markdown 与 Word，无需调用 VLM API...")
        try:
            cache_data = json.loads(middle_json_path.read_text(encoding="utf-8"))
            pdf_info = cache_data.get("pdf_info", [])
            
            all_pages_markdown = []
            for page_data in pdf_info:
                page_markdown_lines = []
                for block in page_data.get("blocks", []):
                    block_type = block.get("type", "").lower()
                    
                    if "content" in block:
                        reflowed_content = smart_reflow_markdown(block["content"])
                        
                        # 按排版格式重建 markdown
                        if block_type == "doc_title":
                            page_markdown_lines.append(f"\n\n# {reflowed_content}\n\n")
                        elif block_type == "paragraph_title":
                            page_markdown_lines.append(f"\n\n## {reflowed_content}\n\n")
                        elif block_type == "table":
                            page_markdown_lines.append(f"\n\n{reflowed_content}\n\n")
                        elif block_type == "abstract":
                            page_markdown_lines.append(f"\n\n> **Abstract** — {reflowed_content}\n\n")
                        elif block_type == "algorithm":
                            page_markdown_lines.append(f"\n\n```\n{reflowed_content}\n```\n\n")
                        elif block_type == "figure_title":
                            page_markdown_lines.append(f"\n\n*{reflowed_content}*\n\n")
                        else:
                            page_markdown_lines.append(f"\n\n{reflowed_content}\n\n")
                    elif "image_path" in block:
                        # 图片块处理
                        image_path = block.get("image_path", "")
                        page_markdown_lines.append(f"\n\n![{block_type}]({image_path})\n\n")
                
                # 拼接单页的 blocks 为单页 Markdown
                page_md = "\n\n".join(page_markdown_lines)
                all_pages_markdown.append(page_md)
                
            # 缝合所有页面并保存
            final_md_content = "\n\n\\newpage\n\n".join(all_pages_markdown)
            output_md_path = output_dir / "final_output.md"
            output_md_path.parent.mkdir(exist_ok=True, parents=True)
            output_md_path.write_text(final_md_content, encoding="utf-8")
            print(f"✅ 最终 Markdown 已保存至: {output_md_path}")
            
            # 调用 Pandoc 转换为 Word
            print("🔄 正在通过 Pandoc 导出 Word 文档...")
            docx_file = convert_file_with_pandoc(output_md_path, "docx")
            if docx_file:
                print(f"🎉 转换成功！Word 文档已输出至: {docx_file.absolute()}")
            else:
                print("⚠️ 提示: Pandoc 转换失败，请确认系统已安装 Pandoc 并配置在环境变量中。")
            return
            
        except Exception as e:
            print(f"⚠️ 读取缓存中介 JSON 失败: {e}，将重新运行完整管线。")

    # 1. 判断如果是 PDF，先渲染成图片
    if input_path.suffix.lower() == ".pdf":
        temp_img_dir = output_dir / "temp_images"
        print(f"🔄 正在将 PDF {input_path} 转换为临时图片...")
        convert_pdf_to_images(input_path, temp_img_dir, dpi=300)
        img_files = sorted(temp_img_dir.glob("page_*.png"))
    else:
        img_files = [input_path]
        
    # 2. 依次运行 Pipeline 获取 Markdown 并缝合
    all_pages_markdown = []
    all_pages_middle_data = []
    for idx, img_path in enumerate(img_files):
        print(f"🧠 正在分析并识别页面: {img_path.name}...")
        # 正确解包返回的 (page_md, page_middle_data)
        page_md, page_middle_data = run_pipeline_flow(str(img_path), str(output_dir), page_idx=idx)
        
        # 3. 对单页内容运行 smart_reflow_markdown 优化换行
        reflowed_md = smart_reflow_markdown(page_md)
        all_pages_markdown.append(reflowed_md)
        all_pages_middle_data.append(page_middle_data)
        
    # 4. 缝合拼接所有页面并保存
    final_md_content = "\n\n\\newpage\n\n".join(all_pages_markdown)
    output_md_path = output_dir / "final_output.md"
    output_md_path.parent.mkdir(exist_ok=True, parents=True)
    output_md_path.write_text(final_md_content, encoding="utf-8")
    print(f"✅ 最终 Markdown 已保存至: {output_md_path}")
    
    # 💾 保存中介 JSON 归档数据实现持久化缓存
    middle_json = {"pdf_info": all_pages_middle_data}
    middle_json_path.parent.mkdir(exist_ok=True, parents=True)
    middle_json_path.write_text(json.dumps(middle_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 中介 JSON 缓存已保存至: {middle_json_path}")
    
    # 5. 调用 Pandoc 转换为 Word (仅在系统存在 pandoc 时生效，不为硬断言)
    print("🔄 正在通过 Pandoc 导出 Word 文档...")
    docx_file = convert_file_with_pandoc(output_md_path, "docx")
    if docx_file:
        print(f"🎉 转换成功！Word 文档已输出至: {docx_file.absolute()}")
    else:
        print("⚠️ 提示: Pandoc 转换失败，请确认系统已安装 Pandoc 并配置在环境变量中。")

if __name__ == "__main__":
    main()
