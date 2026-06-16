import numpy as np
from PIL import Image, ImageDraw
from pipeline.deskew import detect_skew_angle, rotate_image

def test_deskew_logic():
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(50, 100), (350, 126)], fill=(0, 0, 0), width=3)
    draw.line([(50, 200), (350, 226)], fill=(0, 0, 0), width=3)
    
    angle = detect_skew_angle(img)
    assert 4.0 <= angle <= 6.0, f"Detected angle {angle} is out of range!"
    
    rotated_img = rotate_image(img, -angle)
    post_angle = detect_skew_angle(rotated_img)
    assert abs(post_angle) < 1.0, f"Post deskew angle {post_angle} is too large!"
