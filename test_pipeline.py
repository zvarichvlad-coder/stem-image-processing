import unittest
import numpy as np
from src.filters import to_gray

class TestFilters(unittest.TestCase):
    def test_to_gray(self):
        test_img = np.ones((3, 3, 3), dtype=np.uint8) * 100
        res = to_gray(test_img)
        self.assertEqual(res.shape, (3, 3))

if __name__ == "__main__":
    unittest.main()