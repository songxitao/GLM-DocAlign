import unittest
from pipeline.xycut import sort_boxes_by_xy_cut

class TestXYCut(unittest.TestCase):
    def test_xycut_sorting(self):
        # 原版双栏测试
        boxes = [
            {"coords": [220, 15, 400, 95], "label": "text", "id": "C"},
            {"coords": [10, 120, 190, 200], "label": "text", "id": "B"},
            {"coords": [220, 110, 400, 210], "label": "text", "id": "D"},
            {"coords": [10, 10, 190, 100], "label": "text", "id": "A"}
        ]
        
        sorted_indices = sort_boxes_by_xy_cut(boxes)
        sorted_ids = [boxes[idx]["id"] for idx in sorted_indices]
        self.assertEqual(sorted_ids, ["A", "B", "C", "D"], f"XY-Cut sorted order {sorted_ids} is wrong!")

    def test_xycut_single_column_table(self):
        # 测试单栏含有 table 的账单，应该从上到下按 Y 轴切分
        # A (大标题偏右): 200, 10
        # B (表格整行): 10, 100
        # C (签字偏左): 10, 300
        # D (签字偏右): 200, 290
        boxes = [
            {"coords": [200, 10, 350, 50], "label": "doc_title", "id": "A"},
            {"coords": [10, 100, 400, 250], "label": "table", "id": "B"},
            {"coords": [10, 300, 150, 340], "label": "text", "id": "C"},
            {"coords": [200, 290, 350, 330], "label": "text", "id": "D"}
        ]
        
        sorted_indices = sort_boxes_by_xy_cut(boxes)
        sorted_ids = [boxes[idx]["id"] for idx in sorted_indices]
        self.assertEqual(sorted_ids, ["A", "B", "C", "D"], f"Table ledger sorted order {sorted_ids} is wrong!")

if __name__ == '__main__':
    unittest.main()

