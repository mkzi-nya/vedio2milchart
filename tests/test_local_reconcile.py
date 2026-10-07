import unittest

from local_reconcile import align_weak_heads, reconcile, _observed_tail_contact


class CounterTimingRefinementTests(unittest.TestCase):
    def test_joint_assignment_resolves_overlapping_missed_duplicate_and_short_tail_events(self):
        events = [
            {"time": .125, "duration": 0, "isFake": False},
            {"time": .35, "duration": .30, "isFake": False,
             "ringContactSupport": 1.0, "ringHeadSupport": 0.0,
             "headHitSupport": .35, "targetTrack": 2,
             "evidence": "dynamic-target-intersection"},
            {"time": .90, "duration": .30, "isFake": True,
             "ringContactSupport": .65, "ringHeadSupport": 0.0,
             "headHitSupport": .2, "evidence": "full-screen-decorative-track",
             "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
             "holdEndEvidence": "stationary-capsule-tail-and-disappearance",
             "path": [[.7 + i * .03, 240, 80 + i * 15] for i in range(10)]},
            {"time": 1.20, "duration": .25, "isFake": False,
             "track": 77, "x": 300, "ringContactSupport": .9,
             "ringHeadSupport": .9, "headHitSupport": .1,
             "evidence": "full-screen-decorative-track",
             "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
             "holdEndEvidence": "stationary-capsule-tail-and-disappearance"},
            {"time": 1.20, "duration": .25, "isFake": False,
             "track": 77, "x": 302, "ringContactSupport": .9,
             "ringHeadSupport": .9, "headHitSupport": .3,
             "evidence": "occluded-independent-pass",
             "bodyEvidence": "uncropped-bright-capsule-with-overlap-holes",
             "holdEndEvidence": "stationary-capsule-tail-and-disappearance"},
            {"time": 1.45, "duration": 0, "isFake": True,
             "ringContactSupport": 1.0, "ringHeadSupport": 1.0,
             "targetTrack": 9, "evidence": "independent-faint-source-approach",
             "path": [[1.1 + i * .03, 250 + i * 10, 300 + i * 20] for i in range(10)]},
        ]
        counts = [1, 2, 2, 3, 5, 7, 7]
        cumulative = [{"time": (i + 1) / 4, "videoCombo": combo}
                      for i, combo in enumerate(counts)]

        result = reconcile(events, {"cumulative": cumulative})

        self.assertEqual(result["remaining"], 0)
        self.assertEqual(events[1]["duration"], 0.0)
        self.assertFalse(events[2]["isFake"])
        self.assertNotEqual(events[3]["isFake"], events[4]["isFake"])
        self.assertFalse(events[5]["isFake"])

    def test_observed_hold_tail_prevents_counter_fit_from_collapsing_the_hold(self):
        hold = {
            "time": 1.05,
            "duration": .02,
            "isFake": False,
            "visualOnly": False,
            "targetTrack": None,
            "rotation": 90,
            "judgeY": 593 / 720,
            "holdEndEvidence": "stationary-capsule-tail-and-disappearance",
            "bodyEvidence": "full-resolution-blue-capsule-cap-pair",
            "ringContactSupport": .94,
            "ringHeadSupport": .90,
            "sourceContactTime": 1.05,
            "path": [[1.0, 640, 500], [1.05, 640, 570]],
            "tailPath": [
                [.900, 640, 245], [.933, 640, 280], [.967, 640, 315],
                [1.000, 640, 350], [1.033, 640, 385], [1.067, 640, 420],
                [1.100, 640, 455], [1.133, 640, 490], [1.166, 640, 525],
            ],
        }
        tap = {
            "time": 1.06,
            "duration": 0,
            "isFake": False,
            "sourceContactTime": 1.06,
            "targetTrack": 4,
            "ringContactSupport": 1.0,
            "ringHeadSupport": 1.0,
        }
        tail_contact = _observed_tail_contact(hold)
        self.assertGreater(tail_contact, 1.22)
        cumulative = [
            {"time": 1.0, "videoCombo": 0},
            {"time": 1.1, "videoCombo": 2},
            {"time": 1.2, "videoCombo": 2},
            {"time": 1.3, "videoCombo": 3},
        ]

        result = reconcile([hold, tap], {"cumulative": cumulative})

        self.assertEqual(result["changed"], 1)
        self.assertAlmostEqual(hold["time"] + hold["duration"],
                               tail_contact, places=4)
        self.assertEqual(result["remaining"], 0)

    def test_rotating_or_target_tracked_holds_do_not_use_static_tail_geometry(self):
        event = {
            "time": 1.0,
            "duration": .03,
            "isFake": False,
            "rotation": 90,
            "judgeY": .8222,
            "targetTrack": 12,
            "holdEndEvidence": "stationary-capsule-tail-and-disappearance",
            "tailPath": [[1 + i / 30, 640, 300 + i * 20] for i in range(12)],
        }
        self.assertIsNone(_observed_tail_contact(event))
        event["targetTrack"] = None
        event["holdRotationPath"] = [[i / 30, angle] for i, angle in enumerate([80, 85, 90, 96])]
        self.assertIsNone(_observed_tail_contact(event))

    def test_supported_tap_moves_at_most_one_frame_to_counter_transition(self):
        events = [{
            "time": 1.0 + 1 / 30,
            "duration": 0.0,
            "isFake": False,
            "evidence": "dynamic-target-intersection",
            "targetTrack": 7,
            "ringContactSupport": 0.35,
            "ringHeadSupport": 0.1,
            "headHitSupport": 0.2,
            "path": [[.8, 640, 500], [.9, 640, 530], [1.0, 640, 570], [1.03, 640, 590]],
        }]
        observations = [
            {"time": 1.0, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 1 / 30, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 2 / 30, "combo": 1, "error": .01, "margin": .1},
            {"time": 1.0 + 3 / 30, "combo": 1, "error": .01, "margin": .1},
        ]

        result = align_weak_heads(events, observations, fps=30,
                                  max_shift=1 / 30, min_improvement=.75)

        self.assertEqual(result["changed"], 1)
        self.assertAlmostEqual(events[0]["time"], 1.0 + 2 / 30, places=5)
        self.assertAlmostEqual(abs(events[0]["counterTimingAdjustment"]), 1 / 30,
                               delta=1e-5)

    def test_hold_head_and_release_are_not_moved_by_tap_refinement(self):
        events = [{
            "time": 1.0,
            "duration": .4,
            "isFake": False,
            "evidence": "dynamic-target-intersection",
            "targetTrack": 7,
            "ringContactSupport": .3,
            "ringHeadSupport": .1,
            "headHitSupport": .2,
            "path": [[.8, 640, 500], [.9, 640, 530], [1.0, 640, 570], [1.03, 640, 590]],
        }]
        observations = [
            {"time": .95, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.05, "combo": 1, "error": .01, "margin": .1},
            {"time": 1.4, "combo": 2, "error": .01, "margin": .1},
        ]

        result = align_weak_heads(events, observations, fps=30,
                                  max_shift=1 / 30, min_improvement=.5)

        self.assertEqual(result["changed"], 0)
        self.assertEqual(events[0]["time"], 1.0)
        self.assertAlmostEqual(events[0]["time"] + events[0]["duration"], 1.4)

    def test_counter_only_hollow_candidate_is_not_timed_from_combo_alone(self):
        events = [{
            "time": 1.0 + 1 / 30,
            "duration": 0.0,
            "isFake": False,
            "evidence": "faint-hollow-moving-decoration",
            "path": [[.8, 640, 500], [.9, 640, 530], [1.0, 640, 570], [1.03, 640, 590]],
        }]
        observations = [
            {"time": 1.0, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 1 / 30, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 2 / 30, "combo": 1, "error": .01, "margin": .1},
        ]

        result = align_weak_heads(events, observations, fps=30,
                                  max_shift=1 / 30, min_improvement=.5)

        self.assertEqual(result["changed"], 0)
        self.assertAlmostEqual(events[0]["time"], 1.0 + 1 / 30)

    def test_contact_spacing_timing_is_preserved(self):
        events = [{
            "time": 1.0 + 1 / 30,
            "duration": 0.0,
            "isFake": False,
            "evidence": "independent-faint-source-approach",
            "sourceTimingAdjustment": -1 / 30,
            "ringContactSupport": .65,
            "path": [[.8, 640, 500], [.9, 640, 530], [1.0, 640, 570], [1.03, 640, 590]],
        }]
        observations = [
            {"time": 1.0, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 1 / 30, "combo": 0, "error": .01, "margin": .1},
            {"time": 1.0 + 2 / 30, "combo": 1, "error": .01, "margin": .1},
        ]

        result = align_weak_heads(events, observations, fps=30,
                                  max_shift=1 / 30, min_improvement=.5)

        self.assertEqual(result["changed"], 0)
        self.assertAlmostEqual(events[0]["time"], 1.0 + 1 / 30)


if __name__ == "__main__":
    unittest.main()
