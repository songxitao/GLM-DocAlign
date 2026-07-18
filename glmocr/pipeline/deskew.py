import cv2
import numpy as np
from PIL import Image

def detect_skew_angle(image: Image.Image) -> float:
    open_cv_image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=10)
    
    if lines is None:
        return 0.0
        
    angles = []
    for line in lines:
        coords = line.squeeze()
        if coords.size < 4:
            continue
        x1, y1, x2, y2 = coords[:4]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if -15 < angle < 15:
            angles.append(angle)
            
    if not angles:
        return 0.0
    return float(np.median(angles))

def rotate_image(image: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 0.1:
        return image
    open_cv_image = np.array(image.convert("RGB"))
    (h, w) = open_cv_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return Image.fromarray(rotated)
