import unittest

import numpy as np

from counter_frames_v4 import template_errors


class CounterTemplateScoreTests(unittest.TestCase):
    def test_dilated_match_above_raw_pixel_count_does_not_wrap(self):
        # The first candidate's dilated hit count is higher than the raw crop
        # count, a normal case for a clean glyph. Unsigned subtraction used to
        # turn the overlap error into an enormous value and select the wrong
        # digit template.
        totals = np.asarray([110, 130], dtype=np.uint64)
        hits = np.asarray([108, 90], dtype=np.uint64)

        errors = template_errors(totals, hits, observed_pixels=100)

        self.assertTrue(np.isfinite(errors).all())
        self.assertLess(errors[0], errors[1])
        self.assertAlmostEqual(float(errors[0]), 2 / 170)


if __name__ == "__main__":
    unittest.main()
