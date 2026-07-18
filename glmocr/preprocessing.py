# preprocessing.py
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

def convert_pdf_to_images(pdf_path: Path, output_dir: Path, target_pages: list = None, dpi: int = 300) -> Path:
    """
    将PDF文件的指定页面转换为PNG图片。
    
    Args:
        pdf_path (Path): 输入的PDF文件路径。
        output_dir (Path): 保存图片的输出文件夹。
        target_pages (list): 要处理的页码列表（0-based索引）。如果为None，则处理所有页面。
        dpi (int): 渲染图片的分辨率。
    """
    print(f"--- 预处理开始：正在将PDF '{pdf_path.name}' 转换为图片... ---")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    # 1. 确定要处理哪些页
    if target_pages:
        # 过滤掉超出范围的页码，并去重排序
        valid_pages = sorted(list(set([p for p in target_pages if 0 <= p < total_pages])))
        if not valid_pages:
            print(f"⚠️ 警告：指定的页码均超出范围 (总页数: {total_pages})，将默认处理所有页面。")
            pages_to_render = range(total_pages)
        else:
            pages_to_render = valid_pages
            print(f"ℹ️ 仅处理选定页面: {[p+1 for p in pages_to_render]}") # 打印给用户看时+1
    else:
        pages_to_render = range(total_pages)

    # 2. 循环渲染
    # 使用tqdm显示进度条
    for i in tqdm(pages_to_render, desc="渲染PDF页面", unit="页"):
        page = doc[i] # 获取指定页
        pix = page.get_pixmap(dpi=dpi)
        # 文件名依然使用 page_00001.png 这种格式，保持真实页码对应
        # i+1 确保文件名里的页码是人类可读的（从1开始）
        image_path = output_dir / f"page_{str(i+1).zfill(5)}.png"
        pix.save(image_path)
        
    print(f"--- 预处理完成：共生成 {len(pages_to_render)} 张图片 ---")
    return output_dir