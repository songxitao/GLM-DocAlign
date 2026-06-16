import numpy as np
from PIL import Image, ImageDraw

def crop_and_mask(image: Image.Image, boxes: list, target_idx: int) -> Image.Image:
    box = boxes[target_idx]['coords']
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin
    
    # Dynamic padding calculation (Safe Mode)
    pad_x = max(6, int(width * 0.02))
    pad_y = max(6, int(height * 0.03))
    
    img_w, img_h = image.size
    px1 = max(0, xmin - pad_x)
    py1 = max(0, ymin - pad_y)
    px2 = min(img_w, xmax + pad_x)
    py2 = min(img_h, ymax + pad_y)
    
    # Crop sub-image
    cropped = image.crop((px1, py1, px2, py2))
    draw = ImageDraw.Draw(cropped)
    
    # Mask out neighboring elements
    for idx, other in enumerate(boxes):
        if idx == target_idx:
            continue
        ox1, oy1, ox2, oy2 = other['coords']
        
        # Calculate overlap relative to padded bounds
        ix1 = max(px1, ox1)
        iy1 = max(py1, oy1)
        ix2 = min(px2, ox2)
        iy2 = min(py2, oy2)
        
        if ix2 > ix1 and iy2 > iy1:
            local_x1 = ix1 - px1
            local_y1 = iy1 - py1
            local_x2 = ix2 - px1
            local_y2 = iy2 - py1
            draw.rectangle([local_x1, local_y1, local_x2, local_y2], fill=(255, 255, 255))
            
    # Resize upscale if height is less than 64px
    new_h = py2 - py1
    if new_h < 64:
        ratio = 128.0 / new_h
        new_w = int((px2 - px1) * ratio)
        cropped = cropped.resize((new_w, 128), Image.Resampling.BICUBIC)
        
    return cropped
