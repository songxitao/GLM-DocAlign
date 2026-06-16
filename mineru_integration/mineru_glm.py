def map_and_normalize_box(label: str, box: list[float], width: int, height: int) -> tuple[str, list[float]]:
    # Map label: 'formula' -> 'equation', 'figure' -> 'image', others to lowercase
    label_lower = label.lower()
    if label_lower == 'formula':
        mapped_label = 'equation'
    elif label_lower == 'figure':
        mapped_label = 'image'
    else:
        mapped_label = label_lower

    # Normalize box coordinates: xmin, xmax by width; ymin, ymax by height
    # Bounding box is [x1, y1, x2, y2]. Let's handle cases where width or height is 0 or negative.
    w = max(1, width)
    h = max(1, height)

    x1 = max(0.0, min(1.0, float(box[0]) / w))
    y1 = max(0.0, min(1.0, float(box[1]) / h))
    x2 = max(0.0, min(1.0, float(box[2]) / w))
    y2 = max(0.0, min(1.0, float(box[3]) / h))

    return mapped_label, [x1, y1, x2, y2]
