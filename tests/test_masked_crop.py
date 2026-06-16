import numpy as np
from PIL import Image
from pipeline.masked_crop import crop_and_mask

def test_masked_crop_logic():
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    
    boxes = [
        {"coords": [20, 20, 100, 100], "label": "text"},
        {"coords": [80, 80, 180, 180], "label": "text"}
    ]
    
    cropped = crop_and_mask(img, boxes, target_idx=0)
    
    assert cropped.size[1] >= 64
    cropped_np = np.array(cropped)
    # The right-bottom overlapping area in cropped image should be filled with white (255)
    pixel = cropped_np[cropped.size[1] - 10, cropped.size[0] - 10]
    assert np.mean(pixel) > 240

def test_masked_crop_upscale():
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    boxes = [
        {"coords": [20, 20, 100, 50], "label": "text"},
        {"coords": [80, 30, 180, 180], "label": "text"}
    ]
    cropped = crop_and_mask(img, boxes, target_idx=0)
    # The height should be upscaled to 128 since original cropped height (42) < 64
    assert cropped.size[1] == 128
    
    cropped_np = np.array(cropped)
    # After upscale, the vertical overlap range [16, 42] becomes [48.7, 128]
    # So the middle y (64) is within the white masked area
    pixel = cropped_np[cropped.size[1] // 2, cropped.size[0] - 15]
    assert np.mean(pixel) > 240
