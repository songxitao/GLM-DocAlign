import os
import sys
import asyncio
from pathlib import Path
from preprocessing import convert_pdf_to_images
from postprocessing import smart_reflow_markdown, convert_file_with_pandoc
from pipeline.orchestrator import run_pipeline_flow

def main():
    if len(sys.argv) < 2:
        print("📢 用法: python run_pipeline.py <PDF路径或图片路径> [输出目录]")
        sys.exit(1)
        
    input_path = Path(sys.argv[1])
    
    # 智能命名：使用输入文件名（去后缀的 stem）命名输出目录，实现更好的文件归档
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2]) / input_path.stem
    else:
        output_dir = Path("ocr_output") / input_path.stem
    
    if not input_path.exists():
        print(f"❌ 找不到输入文件: {input_path}")
        sys.exit(1)
        
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
    for img_path in img_files:
        print(f"🧠 正在分析并识别页面: {img_path.name}...")
        page_md = run_pipeline_flow(str(img_path), str(output_dir))
        
        # 3. 对单页内容运行 smart_reflow_markdown 优化换行
        reflowed_md = smart_reflow_markdown(page_md)
        all_pages_markdown.append(reflowed_md)
        
    # 4. 缝合拼接所有页面并保存
    final_md_content = "\n\n\\newpage\n\n".join(all_pages_markdown)
    output_md_path = output_dir / "final_output.md"
    output_md_path.parent.mkdir(exist_ok=True, parents=True)
    output_md_path.write_text(final_md_content, encoding="utf-8")
    print(f"✅ 最终 Markdown 已保存至: {output_md_path}")
    
    # 5. 调用 Pandoc 转换为 Word (仅在系统存在 pandoc 时生效，不为硬断言)
    print("🔄 正在通过 Pandoc 导出 Word 文档...")
    docx_file = convert_file_with_pandoc(output_md_path, "docx")
    if docx_file:
        print(f"🎉 转换成功！Word 文档已输出至: {docx_file.absolute()}")
    else:
        print("⚠️ 提示: Pandoc 转换失败，请确认系统已安装 Pandoc 并配置在环境变量中。")

if __name__ == "__main__":
    main()
