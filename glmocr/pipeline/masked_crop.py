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
    
    # Local coordinates of the target core box
    cx1 = xmin - px1
    cy1 = ymin - py1
    cx2 = xmax - px1
    cy2 = ymax - py1
    
    pw = px2 - px1
    ph = py2 - py1
    
    target_area = width * height
    
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
            
            other_area = (ox2 - ox1) * (oy2 - oy1)
            
            # If the other box is significantly smaller (e.g. < 50% area of target),
            # it means this is a small box (e.g. text) inside a large box (e.g. table).
            # We wipe out the small box region inside the large box to avoid double recognition.
            if other_area < target_area * 0.5:
                cx_ix1 = max(local_x1, cx1)
                cy_iy1 = max(local_y1, cy1)
                cx_ix2 = min(local_x2, cx2)
                cy_iy2 = min(local_y2, cy2)
                if cx_ix2 > cx_ix1 and cy_iy2 > cy_iy1:
                    draw.rectangle([cx_ix1, cy_iy1, cx_ix2, cy_iy2], fill=(255, 255, 255))
            
            # Subdivide mask into 4 quadrants around the core box to mask the padding areas
            # 1. Left padding region: (0, 0, cx1, ph)
            lx1, ly1, lx2, ly2 = max(local_x1, 0), max(local_y1, 0), min(local_x2, cx1), min(local_y2, ph)
            if lx2 > lx1 and ly2 > ly1:
                draw.rectangle([lx1, ly1, lx2, ly2], fill=(255, 255, 255))
            
            # 2. Right padding region: (cx2, 0, pw, ph)
            rx1, ry1, rx2, ry2 = max(local_x1, cx2), max(local_y1, 0), min(local_x2, pw), min(local_y2, ph)
            if rx2 > rx1 and ry2 > ry1:
                draw.rectangle([rx1, ry1, rx2, ry2], fill=(255, 255, 255))
                
            # 3. Top padding region: (0, 0, pw, cy1)
            tx1, ty1, tx2, ty2 = max(local_x1, 0), max(local_y1, 0), min(local_x2, pw), min(local_y2, cy1)
            if tx2 > tx1 and ty2 > ty1:
                draw.rectangle([tx1, ty1, tx2, ty2], fill=(255, 255, 255))
                
            # 4. Bottom padding region: (0, cy2, pw, ph)
            bx1, by1, bx2, by2 = max(local_x1, 0), max(local_y1, cy2), min(local_x2, pw), min(local_y2, ph)
            if bx2 > bx1 and by2 > by1:
                draw.rectangle([bx1, by1, bx2, by2], fill=(255, 255, 255))
            
    # Resize upscale if height is less than 64px
    new_h = py2 - py1
    if new_h < 64:
        ratio = 128.0 / new_h
        new_w = int((px2 - px1) * ratio)
        cropped = cropped.resize((new_w, 128), Image.Resampling.BICUBIC)
        
    return cropped
