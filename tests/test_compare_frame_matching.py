import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_fullres import find_render_frame, geometry_diff_image


class RenderFrameMatchingTests(unittest.TestCase):
    def test_matches_full_precision_node_timestamp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wanted = root / "milplay-90.53331996.png"
            wanted.touch()
            path, matched_time = find_render_frame(root, 90.533319960, 0.000001)

            self.assertEqual(path, wanted)
            self.assertAlmostEqual(matched_time, 90.53331996)

    def test_applies_explicit_render_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wanted = root / "milplay-121.766333.png"
            wanted.touch()
            path, _ = find_render_frame(root, 121.733, 0.033333)

            self.assertEqual(path, wanted)

    def test_does_not_silently_select_a_neighbor_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "milplay-90.500.png").touch()

            with self.assertRaises(FileNotFoundError):
                find_render_frame(root, 90.533319960, 0.00001)

    def test_geometry_diff_separates_missing_and_extra_strokes(self):
        source = np.zeros((24, 24), dtype=bool)
        rendered = np.zeros_like(source)
        ignored = np.zeros_like(source)
        source[5, 5:10] = True
        rendered[6, 5:10] = True  # within the four-pixel raster tolerance
        source[16, 16] = True
        rendered[22, 22] = True

        result = geometry_diff_image(source, rendered, ignored)

        self.assertTrue(np.any(np.all(result == (90, 220, 90), axis=2)))
        self.assertEqual(tuple(result[16, 16]), (255, 64, 64))
        self.assertEqual(tuple(result[22, 22]), (40, 220, 255))


if __name__ == "__main__":
    unittest.main()
