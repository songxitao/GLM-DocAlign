# postprocessing.py (The Complete Fixed Version)
# 包含：smart_reflow_markdown, stitch_pages, merge_markdown_files, 
# convert_file_with_pandoc, process_grounded_markdown, create_diagnosis_image

import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import shutil
import subprocess

try:
    from markdownify import markdownify as md
except ImportError:
    print("!!!!!! 关键库缺失 !!!!!!")
    print("请先在你的环境中运行: pip install markdownify")
    exit(1)

# --- 辅助判断函数 ---
def is_cjk(char):
    """判断字符是否为中日韩文字"""
    if not char: return False
    return '\u4e00' <= char <= '\u9fff'

def is_sentence_end(char):
    """判断是否为句末标点 (中英文)"""
    return char in {'.', '!', '?', '。', '！', '？', '…', '”', '"'}

# 注册连字符字符集
LINE_END_HYPHEN_CHARS = "-\u00ad\u2010\u2011\u2043"

def full_to_half_exclude_marks(text: str) -> str:
    """
    仅转换 FF21-FF3A (全角大写 A-Z), FF41-FF5A (全角小写 a-z), FF10-FF19 (全角数字 0-9) 范围内的字符。
    其他字符（如公式符号 ＋, ＝，标点 ！）保持原样。
    """
    res = []
    for char in text:
        code = ord(char)
        if (0xFF10 <= code <= 0xFF19) or (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A):
            res.append(chr(code - 0xfee0))
        else:
            res.append(char)
    return "".join(res)

# --- 1. 单页重排 (保守版 + 中文优化) ---
def smart_reflow_markdown(text):
    """
    保守版重排：
    1. 保护表格、标题、列表不被合并。
    2. 英文：上一行非句号 + 下一行小写 -> 合并。
    3. 中文：上一行汉字 + 下一行汉字 -> 合并。
    """
    lines = text.split('\n')
    new_lines = []
    buffer = ""
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        
        # 1. 代码块保护 (绝对不动)
        if stripped.startswith("```"):
            if buffer: new_lines.append(buffer); buffer = ""
            new_lines.append(line)
            in_code_block = not in_code_block
            continue
        if in_code_block:
            new_lines.append(line)
            continue
            
        # 2. 空行处理 (视为段落结束)
        if not stripped:
            if buffer: new_lines.append(buffer); buffer = ""
            new_lines.append(line)
            continue

        # 3. 敏感结构识别 (标题、列表、引用、表格、LaTeX公式)
        # ✨ 关键：遇到 | 开头，认为是表格，立即停止合并！
        is_sensitive = (
            stripped.startswith(("#", "- ", "* ", "> ", "|", "<table", "</table", "$$")) or
            re.match(r'^\d+\.', stripped) or  # 数字列表 1.
            re.match(r'^(Figure|Fig\.|Table)\s*\d+', stripped) # 图注
        )

        if is_sensitive:
            if buffer: new_lines.append(buffer); buffer = ""
            new_lines.append(line)
            continue

        # --- 核心逻辑：保守合并 ---
        if buffer:
            prev_char = buffer.strip()[-1] if buffer.strip() else ""
            curr_char = stripped[0] if stripped else ""
            
            should_merge = False
            merge_sep = " " # 默认英文空格

            # 情况 B: 连字符合并 (如 connec- tion, 能够拼合 5 种连字符)
            if prev_char in LINE_END_HYPHEN_CHARS:
                should_merge = True
                merge_sep = ""
                # 去除行尾所有的连字符字符，同时保留前导空白
                stripped_buf = buffer.strip()
                while stripped_buf and stripped_buf[-1] in LINE_END_HYPHEN_CHARS:
                    stripped_buf = stripped_buf[:-1]
                leading_spaces = buffer[:len(buffer) - len(buffer.lstrip())]
                buffer = leading_spaces + stripped_buf

            # 情况 A: 英文合并 (非句号结尾 + 小写开头)
            elif not is_sentence_end(prev_char) and curr_char.islower():
                should_merge = True

            # 情况 C: ✨ 中文合并 (汉字结尾 + 汉字开头)
            elif is_cjk(prev_char) and is_cjk(curr_char):
                should_merge = True
                merge_sep = "" # 中文不需要空格

            if should_merge:
                buffer += merge_sep + stripped
            else:
                # 不满足合并条件，输出上一段，开始新一段
                new_lines.append(buffer)
                buffer = line
        else:
            buffer = line

    if buffer: new_lines.append(buffer)
    return "\n".join(new_lines)

# --- 2. 跨页缝合 (增强中文支持) ---
def stitch_pages(pages_content):
    """
    将每一页的内容缝合在一起。
    支持：
    1. 英文跨页：connec- tion -> connection
    2. 英文跨页：sentence start -> sentence start
    3. ✨ 中文跨页：第一页末无标点 -> 直接拼接到第二页首
    """
    final_text = ""
    
    for i, page_text in enumerate(pages_content):
        if not page_text.strip(): continue
        
        if i == 0:
            final_text = page_text
            continue
            
        # 获取连接点字符
        prev_end = final_text.strip()[-1] if final_text.strip() else ""
        curr_start = page_text.strip()[0] if page_text.strip() else ""
        
        # --- 缝合逻辑 ---
        
        # 1. 英文连字符
        if prev_end in LINE_END_HYPHEN_CHARS:
            # 去除前一页末尾所有的连字符
            stripped_text = final_text.rstrip()
            while stripped_text and stripped_text[-1] in LINE_END_HYPHEN_CHARS:
                stripped_text = stripped_text[:-1]
            final_text = stripped_text + page_text.strip()
            
        # 2. 英文句子跨页 (非句号 + 小写)
        elif (prev_end.isalpha() or prev_end == ",") and curr_start.islower():
            final_text += " " + page_text.strip()
            
        # 3. ✨ 中文句子跨页 (汉字结尾 + 汉字开头)
        # 如果上一页以汉字结尾（且不是句号），下一页以汉字开头 -> 直接拼
        elif is_cjk(prev_end) and is_cjk(curr_start) and not is_sentence_end(prev_end):
             final_text += page_text.strip() # 无空格拼接
             
        # 4. 其他情况 (正常分段)
        else:
            final_text += "\n\n" + page_text
            
    return final_text

# --- 3. 合并入口 (你刚刚缺失的部分！) ---
# postprocessing.py

def merge_markdown_files(source_dir: Path, output_file: Path, mode: str = "paginated"):
    print(f"\n--- Post-processing: Merging with mode '{mode}'... ---")
    md_files = sorted(source_dir.glob("page_*.md"))
    if not md_files: return
    
    if mode == "paginated":
        # 分页模式：保留分页符
        separator = '\n\n\\newpage\n\n'
        
        # ✨ 修复逻辑：使用 list 收集内容，最后用 join 连接
        # 这样最后一页后面就不会有多余的 separator 了
        pages_content = []
        for md_file in tqdm(md_files, desc="Merging Paginated"):
            pages_content.append(md_file.read_text(encoding="utf-8").strip()) # strip() 去除首尾多余空行
            
        full_text = separator.join(pages_content)
        output_file.write_text(full_text, encoding="utf-8")
        
    else:
        # 流式模式：重排 + 缝合
        processed_pages = []
        for md_file in tqdm(md_files, desc="Reflowing Pages"):
            raw_text = md_file.read_text(encoding="utf-8")
            # 1. 单页重排
            reflowed_text = smart_reflow_markdown(raw_text)
            processed_pages.append(reflowed_text)
            
        # 2. 跨页缝合
        full_text = stitch_pages(processed_pages)
        output_file.write_text(full_text, encoding="utf-8")

    print(f"--- Merging complete. Saved to '{output_file.name}' ---")

# --- 4. Pandoc 转换 ---
def convert_file_with_pandoc(input_file: Path, output_format: str, use_lua: bool = False):
    if not shutil.which("pandoc"):
        print("\n--- Error: Pandoc not found. ---")
        return None
        
    lua_filter_path = (Path(__file__).parent / "pagebreak.lua").resolve().absolute()
    template_path = (Path(__file__).parent / "my_template_academic.docx").resolve().absolute()
    if not template_path.exists():
        template_path = (Path(__file__).parent / "my_template1.docx").resolve().absolute()
    if not template_path.exists():
        template_path = (Path(__file__).parent / "my_template.docx").resolve().absolute()
        
    # 预处理 HTML 表格
    original_content = input_file.read_text(encoding="utf-8")
    def table_replacer(match):
        return md(match.group(0), heading_style="ATX")
    cleaned_content = re.sub(r"<table.*?>.*?</table>", table_replacer, original_content, flags=re.DOTALL | re.IGNORECASE)
    
    preprocessed_file = input_file.with_name(f"{input_file.stem}_preprocessed.md")
    preprocessed_file.write_text(cleaned_content, encoding="utf-8")

    output_file = input_file.with_suffix(f".{output_format}")
    input_format = "markdown+pipe_tables+tex_math_dollars+tex_math_single_backslash"
    
    command = [
        "pandoc",
        "-f", input_format,
        str(preprocessed_file.name),
        "-o",
        str(output_file.name)
    ]
    
    if template_path.exists():
        command.extend(["--reference-doc", str(template_path)])  # <--- 新增这行！
    else:
        print(f"⚠️ 警告: 找不到模板文件 {template_path}，将使用默认样式。")
        
    if use_lua and lua_filter_path.exists():
        command.insert(1, str(lua_filter_path))
        command.insert(1, "--lua-filter")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", cwd=str(input_file.parent))
        preprocessed_file.unlink()
        return output_file
    except Exception as e:
        print(f"Pandoc error: {e}")
        if preprocessed_file.exists(): preprocessed_file.unlink()
        return None

# --- 5. 核心修复：Grounding 处理 (含暴力清洗) ---

NORMALIZATION_FACTOR = 1000.0

def process_grounded_markdown(md_path: Path, original_image_path: Path, output_dir: Path):
    """
    兼容模式：同时支持 'text[[...]]' 和 '<|ref|>text<|/ref|><|det|>[[...]]<|/det|>'
    """
    images_subdir = output_dir / "images"; images_subdir.mkdir(exist_ok=True)
    content = md_path.read_text(encoding="utf-8")
    original_image = Image.open(original_image_path)
    width_orig, height_orig = original_image.size
    counters = {"image": 1, "table": 1, "chart": 1}
    
    # 正则：支持两种格式
    pattern = re.compile(r"(?:<\|ref\|>)?([a-zA-Z_]+)(?:<\|/ref\|>)?(?:<\|det\|>)?(\[\[(\d+(?:,\s*\d+){3})\]\])(?:<\|/det\|>)?")

    def replacement_func(match):
        tag_type = match.group(1).lower()
        coords_str = match.group(2)
        
        # 提取数字
        nums = re.findall(r"\d+", coords_str)
        if not nums or len(nums) < 4: return ""
        coords = [int(x) for x in nums[:4]] # 确保只取前4个

        clean_tag = None
        if "image" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
            clean_tag = "image"
        elif "table" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
            clean_tag = "table"
        elif "chart" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
            clean_tag = "chart"

        if clean_tag:
            try:
                x1 = int(coords[0] / NORMALIZATION_FACTOR * width_orig)
                y1 = int(coords[1] / NORMALIZATION_FACTOR * height_orig)
                x2 = int(coords[2] / NORMALIZATION_FACTOR * width_orig)
                y2 = int(coords[3] / NORMALIZATION_FACTOR * height_orig)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width_orig, x2), min(height_orig, y2)
                
                if x2 > x1 and y2 > y1:
                    cropped = original_image.crop((x1, y1, x2, y2))
                    idx = counters[clean_tag]
                    image_filename = f"{original_image_path.stem}_{clean_tag}_{idx}.png"
                    cropped.save(images_subdir / image_filename)
                    counters[clean_tag] += 1
                    return f"\n\n![{clean_tag}](images/{image_filename})\n\n"
            except Exception: return ""
        return "" # 删除标签

    new_content = pattern.sub(replacement_func, content)
    
    # 暴力清洗残留
    new_content = new_content.replace("<|ref|>", "").replace("<|/ref|>", "").replace("<|det|>", "").replace("<|/det|>", "")
    garbage_pattern = re.compile(r"[a-zA-Z_]+\[\[.*?\]\]")
    new_content = garbage_pattern.sub("", new_content)
    
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    md_path.write_text(new_content, encoding="utf-8")

def create_diagnosis_image(md_path: Path, original_image_path: Path, output_dir: Path):
    diagnosis_subdir = output_dir / "diagnosis"; diagnosis_subdir.mkdir(exist_ok=True)
    try:
        content = md_path.read_text(encoding="utf-8"); image_copy = Image.open(original_image_path).convert("RGBA"); draw = ImageDraw.Draw(image_copy); width_orig, height_orig = image_copy.size
        colors = {"image": "red", "table": "blue", "chart": "orange", "title": "blue", "sub_title": "purple", "text": "green", "default": "yellow"}
        pattern = re.compile(r"(?:<\|ref\|>)?([a-zA-Z_]+)(?:<\|/ref\|>)?(?:<\|det\|>)?(\[\[(\d+(?:,\s*\d+){3})\]\])(?:<\|/det\|>)?")
        for match in pattern.finditer(content):
            tag_type = match.group(1).lower(); 
            nums = re.findall(r"\d+", match.group(2))
            if len(nums) < 4: continue
            coords = [int(x) for x in nums[:4]]
            try:
                x1 = int(coords[0] / NORMALIZATION_FACTOR * width_orig); y1 = int(coords[1] / NORMALIZATION_FACTOR * height_orig); x2 = int(coords[2] / NORMALIZATION_FACTOR * width_orig); y2 = int(coords[3] / NORMALIZATION_FACTOR * height_orig)
                clean_tag = tag_type
                if "image" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
                    clean_tag = "image"
                elif "table" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
                    clean_tag = "table"
                elif "chart" in tag_type and "caption" not in tag_type and "footnote" not in tag_type:
                    clean_tag = "chart"
                color = colors.get(clean_tag, colors.get(tag_type, colors["default"])); draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            except: continue
        image_copy.save(diagnosis_subdir / f"{original_image_path.stem}_diagnosis.png")
    except Exception: pass