import unittest
import os
import sys

# Ensure the directory of this test file is in sys.path so we can import mineru_glm
sys.path.insert(0, os.path.dirname(__file__))

class TestMineruGlm(unittest.TestCase):
    def test_map_and_normalize_box_formula(self):
        from mineru_vl_utils import map_and_normalize_box
        # Test input pixel coordinates [100, 200, 300, 400], image size 1000x1000
        # formula -> equation, normalized [0.1, 0.2, 0.3, 0.4]
        label, box = map_and_normalize_box('formula', [100, 200, 300, 400], 1000, 1000)
        self.assertEqual(label, 'equation')
        self.assertEqual(box, [0.1, 0.2, 0.3, 0.4])

    def test_map_and_normalize_box_figure(self):
        from mineru_vl_utils import map_and_normalize_box
        # Test input pixel coordinates [100, 200, 300, 400], image size 1000x1000
        # figure -> image, normalized [0.1, 0.2, 0.3, 0.4]
        label, box = map_and_normalize_box('figure', [100, 200, 300, 400], 1000, 1000)
        self.assertEqual(label, 'image')
        self.assertEqual(box, [0.1, 0.2, 0.3, 0.4])

if __name__ == '__main__':
    unittest.main()
