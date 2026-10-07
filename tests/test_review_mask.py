import unittest

import cv2
import numpy as np

from compare_fullres import source_hud_mask


class SourceHudMaskTests(unittest.TestCase):
    def test_native_ui_glyphs_are_masked_without_hiding_crossing_lane(self):
        rgb = np.full((720, 1280, 3), (12, 18, 30), dtype=np.uint8)
        cv2.line(rgb, (160, 0), (160, 719), (240, 246, 255), 6)
        cv2.putText(rgb, "Cloud", (120, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (210, 216, 224), 2, cv2.LINE_AA)

        ignored = source_hud_mask(rgb)

        self.assertTrue(ignored[35:75, 116:244].any())
        self.assertFalse(ignored[60, 160])
        self.assertFalse(ignored[120, 130])

    def test_top_progress_stroke_is_filtered_but_top_edge_is_kept(self):
        rgb = np.full((720, 1280, 3), (12, 18, 30), dtype=np.uint8)
        cv2.line(rgb, (80, 2), (220, 2), (245, 248, 255), 2)
        cv2.line(rgb, (300, 0), (300, 719), (245, 248, 255), 6)

        ignored = source_hud_mask(rgb)

        self.assertTrue(ignored[2, 150])
        self.assertFalse(ignored[2, 300])
        self.assertFalse(ignored[12, 500])

    def test_judgement_strokes_through_top_hud_remain_comparable(self):
        rgb = np.full((720, 1280, 3), (12, 18, 30), dtype=np.uint8)
        cv2.line(rgb, (0, 60), (1279, 60), (240, 246, 255), 3)
        cv2.line(rgb, (0, 35), (600, 90), (240, 246, 255), 3)
        cv2.putText(rgb, "Cloud", (120, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (210, 216, 224), 2, cv2.LINE_AA)

        ignored = source_hud_mask(rgb)

        self.assertTrue(ignored[35:75, 116:244].any())
        self.assertFalse(ignored[60, 300])
        self.assertFalse(ignored[63, 300])


if __name__ == "__main__":
    unittest.main()
