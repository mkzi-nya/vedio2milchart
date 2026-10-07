import unittest

from export_v2 import (
    extend_measured_approach,
    export_js,
    filter_disconnected_rail_fragments,
    fallback_approach_lead,
    moving_hold_anchor,
)


class MeasuredApproachTests(unittest.TestCase):
    def test_drops_offscreen_to_target_teleporting_rail_fragment(self):
        effects = [{
            "kind": "rail",
            "samples": [
                {"time": 110.56665, "x": 2058.9, "y": 2773.2,
                 "rotation": -129.3, "nativeAnchor": True, "anchored": False},
                {"time": 110.59998, "x": -480, "y": -360,
                 "rotation": -306.4, "nativeAnchor": True, "anchored": True},
            ],
        }]

        self.assertEqual(filter_disconnected_rail_fragments(effects), [])

    def test_keeps_continuous_short_rotating_rail(self):
        effects = [{
            "kind": "rail",
            "samples": [
                {"time": 1.0, "x": -20, "y": 50, "rotation": 90,
                 "nativeAnchor": True, "anchored": True},
                {"time": 1.033, "x": -18, "y": 52, "rotation": 77,
                 "nativeAnchor": True, "anchored": True},
            ],
        }]

        self.assertEqual(filter_disconnected_rail_fragments(effects), effects)

    def test_counter_only_contact_does_not_get_a_guessed_slow_fall(self):
        inferred = {"confidence": "inferred", "isFake": False}
        observed = {"confidence": "medium", "isFake": False}

        self.assertEqual(fallback_approach_lead(inferred, []), 0.02)
        self.assertEqual(fallback_approach_lead(observed, []), 0.6)
        self.assertEqual(fallback_approach_lead(inferred, [[0.9, 640, 400]]), 0.6)

    def test_short_pre_hit_track_is_not_extrapolated(self):
        path = [
            [0.98, 640, 560],
            [0.99, 640, 590],
            [1.00, 640, 620],
        ]

        result = extend_measured_approach(path, 1.0)

        self.assertEqual(result, path)
        self.assertEqual(result[0][0], 0.98)

    def test_sustained_measured_fall_gets_a_top_entry(self):
        path = [
            [0.50, 640, 180],
            [0.5333, 640, 212],
            [0.5667, 640, 244],
            [0.60, 640, 276],
            [0.6333, 640, 308],
        ]

        result = extend_measured_approach(path, 1.0)

        self.assertLess(result[0][0], path[0][0])
        self.assertLessEqual(result[0][2], 1)
        self.assertEqual(result[1:], path)

    def test_visual_animation_preload_does_not_retime_chart_hits(self):
        event = {
            "time": 1.0,
            "x": 640,
            "judgeY": 0.82,
            "type": 0,
            "duration": 0,
            "isFake": False,
            "path": [[0.50, 640, 180], [0.533333, 640, 212],
                     [0.566667, 640, 244], [0.60, 640, 276]],
        }

        source = export_js([event])

        self.assertIn("const vt=t=>Math.max(0,Math.round((t-.00002)*1e6)/1e6)", source)
        self.assertIn("m.note(ln[0],b,1.00000,1.00000,0,false,false)", source)

    def test_moving_hold_uses_measured_post_hit_head_when_anchor_is_static(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyPath": [[0.966667, 510, 545], [1.033333, 490, 590],
                         [1.066667, 460, 605], [1.1, 430, 610]],
        }
        static_anchor = [[1.0, 500, 576], [1.2, 500, 576]]

        result = moving_hold_anchor(event, static_anchor)

        self.assertEqual(result[0], [1.0, 500.0, 576.0])
        self.assertEqual(result[1:4], event["bodyPath"][1:])
        self.assertEqual(result[-1], [1.2, 430.0, 610.0])

    def test_trusts_repeated_high_resolution_head_disagreement(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[1.033333, 490, 590], [1.066667, 460, 605],
                         [1.1, 430, 610]],
        }
        ring_anchor = [[1.0, 500, 576], [1.2, 500, 576]]

        result = moving_hold_anchor(event, ring_anchor)

        self.assertEqual(result[0], [1.0, 500.0, 576.0])
        self.assertEqual(result[1:4], event["bodyPath"])
        self.assertEqual(result[-1], [1.2, 430.0, 610.0])

    def test_ignores_one_frame_high_resolution_track_jump(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[1.033333, 490, 590], [1.066667, 460, 635],
                         [1.1, 430, 610]],
        }
        moving_anchor = [[1.0, 500, 576], [1.033333, 490, 590],
                         [1.066667, 460, 605], [1.1, 430, 610],
                         [1.2, 420, 610]]

        result = moving_hold_anchor(event, moving_anchor)

        self.assertIs(result, moving_anchor)

    def test_trusts_three_consecutive_moderate_cap_misses(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[1.033333, 492, 590], [1.066667, 491, 590],
                         [1.1, 492, 590]],
        }
        ring_anchor = [[1.0, 500, 590], [1.2, 500, 590]]

        result = moving_hold_anchor(event, ring_anchor)

        self.assertEqual(result[0], [1.0, 500.0, 576.0])
        self.assertEqual(result[1:4], event["bodyPath"])

    def test_short_hold_uses_strong_cap_sample_just_after_release_frame(self):
        event = {
            "time": 1.0,
            "duration": 0.03,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[1.02, 480, 580], [1.06, 470, 590]],
        }
        ring_anchor = [[1.0, 500, 576], [1.03, 500, 576]]

        result = moving_hold_anchor(event, ring_anchor)

        self.assertEqual(result[0], [1.0, 500.0, 576.0])
        self.assertEqual(result[1], [1.02, 480.0, 580.0])
        self.assertEqual(result[2], [1.03, 470.0, 590.0])

    def test_measured_moving_hold_anchor_is_preserved(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyPath": [[1.033333, 490, 590], [1.066667, 460, 605],
                         [1.1, 430, 610]],
        }
        fitted_anchor = [[1.0, 500, 576], [1.1, 460, 600], [1.2, 420, 610]]

        result = moving_hold_anchor(event, fitted_anchor)

        self.assertIs(result, fitted_anchor)

    def test_repairs_isolated_anchor_spike_confirmed_by_capsule_path(self):
        event = {
            "time": 1.0,
            "duration": 0.2,
            "isFake": False,
            "x": 500,
            "judgeY": 0.8,
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "bodyPath": [[1.033333, 500, 590], [1.066667, 500, 600],
                         [1.1, 500, 610]],
        }
        fitted_anchor = [[1.0, 500, 576], [1.033333, 500, 590],
                         [1.066667, 480, 600], [1.1, 500, 610],
                         [1.2, 500, 610]]

        result = moving_hold_anchor(event, fitted_anchor)

        self.assertEqual(result[2], [1.066667, 500, 600])
        self.assertEqual(fitted_anchor[2], [1.066667, 480, 600])


if __name__ == "__main__":
    unittest.main()
