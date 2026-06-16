from pipeline.xycut import sort_boxes_by_xy_cut

def test_xycut_sorting():
    boxes = [
        {"coords": [220, 15, 400, 95], "label": "text", "id": "C"},
        {"coords": [10, 120, 190, 200], "label": "text", "id": "B"},
        {"coords": [220, 110, 400, 210], "label": "text", "id": "D"},
        {"coords": [10, 10, 190, 100], "label": "text", "id": "A"}
    ]
    
    sorted_indices = sort_boxes_by_xy_cut(boxes)
    sorted_ids = [boxes[idx]["id"] for idx in sorted_indices]
    assert sorted_ids == ["A", "B", "C", "D"], f"XY-Cut sorted order {sorted_ids} is wrong!"
