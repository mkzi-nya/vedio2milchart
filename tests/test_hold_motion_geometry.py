import unittest

import numpy as np

from hold_motion_v6 import capsule_rotations, refine_occluded_hold_orientations


class CapsuleRotationTests(unittest.TestCase):
    def test_keeps_repeated_small_but_visible_tilt(self):
        heads = np.array([[0, 250, 600], [1, 250, 600], [2, 250, 600]], dtype=float)
        angle = np.deg2rad(91.5)
        vector = np.array([np.cos(angle), -np.sin(angle)]) * 600
        tails = np.array([[i, *(heads[i, 1:] + vector)] for i in range(3)])

        result = capsule_rotations(heads, tails)

        np.testing.assert_allclose(result, [91.5, 91.5, 91.5], atol=.02)

    def test_snaps_subpixel_angle_noise_to_vertical(self):
        heads = np.array([[0, 250, 600], [1, 250, 600], [2, 250, 600]], dtype=float)
        angle = np.deg2rad(90.2)
        vector = np.array([np.cos(angle), -np.sin(angle)]) * 600
        tails = np.array([[i, *(heads[i, 1:] + vector)] for i in range(3)])

        result = capsule_rotations(heads, tails)

        np.testing.assert_allclose(result, [90, 90, 90], atol=.02)

    def test_repairs_sustained_obscured_capsule_angle_disagreement(self):
        times = [0.90, 0.933333, 0.966667, 1.0, 1.033333]
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": True,
            "bodyEvidence": "uncropped-bright-capsule-with-overlap-holes",
            "bodyPath": [[t, 400, 400 + 20 * i] for i, t in enumerate(times)],
            "tailPath": [[t, 400, 200 + 20 * i] for i, t in enumerate(times)],
            "holdRotationPath": [[0.9, 76], [1.033333, 76]],
        }

        report = refine_occluded_hold_orientations([event])

        self.assertEqual(report["corrected"], 1)
        self.assertTrue(all(abs(angle - 90) < .02 for _, angle in event["holdRotationPath"][1:-1]))

    def test_uses_capsule_axis_while_one_cap_is_just_outside_frame(self):
        times = [0.90, 0.933333, 0.966667, 1.0, 1.033333]
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": True,
            "bodyEvidence": "uncropped-bright-capsule-with-overlap-holes",
            "bodyPath": [[t, 400, 40 + 20 * i] for i, t in enumerate(times)],
            "tailPath": [[t, 400, -90 + 20 * i] for i, t in enumerate(times)],
            "holdRotationPath": [[0.9, 76], [1.033333, 76]],
        }

        report = refine_occluded_hold_orientations([event])

        self.assertEqual(report["corrected"], 1)
        corrected = [angle for tm, angle in event["holdRotationPath"] if 0.9 <= tm <= 1.033333]
        self.assertTrue(all(abs(angle - 90) < .02 for angle in corrected))

    def test_repairs_persistent_conflict_in_full_resolution_cap_pair(self):
        times = [0.90, 0.933333, 0.966667, 1.0, 1.033333]
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[t, 400, 200 + 20 * i] for i, t in enumerate(times)],
            "tailPath": [[t, 400, 0 + 20 * i] for i, t in enumerate(times)],
            "holdRotationPath": [[0.9, 104], [1.033333, 104]],
        }

        report = refine_occluded_hold_orientations([event])

        self.assertEqual(report["corrected"], 1)
        corrected = [angle for tm, angle in event["holdRotationPath"] if 0.9 <= tm <= 1.033333]
        self.assertTrue(all(abs(angle - 90) < .02 for angle in corrected))

    def test_ignores_single_obscured_cap_angle_disagreement(self):
        times = [0.90, 0.933333, 0.966667, 1.0, 1.033333]
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": True,
            "bodyEvidence": "uncropped-bright-capsule-with-overlap-holes",
            "bodyPath": [[t, 400, 400 + 20 * i] for i, t in enumerate(times)],
            "tailPath": [[t, 400, 200 + 20 * i] for i, t in enumerate(times)],
            "holdRotationPath": [[0.9, 90], [0.933333, 76], [0.966667, 90],
                                 [1.0, 90], [1.033333, 90]],
        }

        report = refine_occluded_hold_orientations([event])

        self.assertEqual(report["corrected"], 0)
        self.assertEqual(event["holdRotationPath"][1][1], 76)

    def test_repairs_three_adjacent_clipped_frames_but_not_an_isolated_frame(self):
        times = [0.90, 0.933333, 0.966667, 1.0, 1.033333]
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": True,
            "bodyEvidence": "uncropped-bright-capsule-with-overlap-holes",
            "bodyPath": [[t, 400, 40 + 20 * i] for i, t in enumerate(times)],
            "tailPath": [[t, 400, -90 + 20 * i] for i, t in enumerate(times)],
            "holdRotationPath": [[t, angle] for t, angle in zip(times, [90, 76, 76, 76, 90])],
        }

        report = refine_occluded_hold_orientations([event])

        self.assertEqual(report["corrected"], 1)
        corrected = [angle for _, angle in event["holdRotationPath"]]
        self.assertTrue(all(abs(angle - 90) < .02 for angle in corrected))


if __name__ == "__main__":
    unittest.main()
