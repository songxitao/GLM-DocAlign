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
    
    # 1. Target core area should be protected and remain black (mean < 10)
    # Coordinate (80, 80) in local space is within target core (xmin=20, ymin=20 -> local xmin=6, local ymin=6)
    # and cx2=86, cy2=86. So (80, 80) is inside target core box and should be black.
    core_pixel = cropped_np[80, 80]
    assert np.mean(core_pixel) < 10
    
    # 2. Right padding area overlaps with the other box (local x > 86, local y < 86, e.g., (90, 80))
    # It should be masked (white)
    right_pad_pixel = cropped_np[80, 90]  # numpy array is indexed [y, x]
    assert np.mean(right_pad_pixel) > 240
    
    # 3. Bottom padding area overlaps with the other box (local x < 86, local y > 86, e.g., (80, 90))
    # It should be masked (white)
    bottom_pad_pixel = cropped_np[90, 80]  # numpy array is indexed [y, x]
    assert np.mean(bottom_pad_pixel) > 240

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

def test_masked_crop_large_target_wipes_small_other():
    # Target (large): [0, 0, 200, 200] (area 40000)
    # Other (small): [80, 80, 100, 100] (area 400) -> < 50% of target area
    img = Image.new("RGB", (300, 300), (0, 0, 0))
    boxes = [
        {"coords": [80, 80, 100, 100], "label": "text"},   # idx = 0 (small)
        {"coords": [0, 0, 200, 200], "label": "table"}       # idx = 1 (large)
    ]
    cropped = crop_and_mask(img, boxes, target_idx=1)
    cropped_np = np.array(cropped)
    
    # Coordinate (85, 85) is inside the small other box
    # px1=0, py1=0, so local coordinates match original coordinates.
    # It should be masked (white)
    pixel = cropped_np[85, 85]
    assert np.mean(pixel) > 240
