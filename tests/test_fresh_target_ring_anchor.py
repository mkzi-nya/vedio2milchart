import unittest

from effect_anchor_v5 import fit


class FreshTargetRingAnchorTests(unittest.TestCase):
    def test_fresh_ring_can_anchor_a_tracked_weak_approach(self):
        fps = 30.0
        samples = [{"time": i / fps, "rings": []} for i in range(61)]
        for index, radius in ((32, 36.0), (33, 45.0), (34, 54.0)):
            samples[index]["rings"] = [{
                "x": 100.0, "y": 100.0, "radius": radius,
                "support": .96, "smallSupport": .92,
            }]
        event = {
            "time": 1.0, "duration": 0.0, "x": 100.0,
            "judgeY": 100 / 720, "type": 1, "speed": 800,
            "isFake": True, "targetTrack": 9,
            "evidence": "independent-faint-source-approach",
            "path": [[.85, 140, 100], [.90, 140, 100], [.95, 140, 100],
                     [1.0, 140, 100], [1.03, 140, 100]],
        }

        fit([event], [], {"fps": fps, "samples": samples})

        self.assertGreaterEqual(event["ringHeadSupport"], .5)
        self.assertGreaterEqual(event["ringContactSupport"], .9)
        self.assertEqual(event["anchorEvidence"], "source-violet-hit-ring-and-matching-circle")

    def test_distant_fresh_ring_does_not_anchor_a_weak_approach(self):
        fps = 30.0
        samples = [{"time": i / fps, "rings": []} for i in range(61)]
        for index, radius in ((32, 36.0), (33, 45.0), (34, 54.0)):
            samples[index]["rings"] = [{
                "x": 180.0, "y": 100.0, "radius": radius,
                "support": .96, "smallSupport": .92,
            }]
        event = {
            "time": 1.0, "duration": 0.0, "x": 100.0,
            "judgeY": 100 / 720, "type": 1, "speed": 800,
            "isFake": True, "targetTrack": 9,
            "evidence": "independent-faint-source-approach",
            "path": [[.85, 140, 100], [.90, 140, 100], [.95, 140, 100],
                     [1.0, 140, 100], [1.03, 140, 100]],
        }

        fit([event], [], {"fps": fps, "samples": samples})

        self.assertEqual(event["ringHeadSupport"], 0.0)
        self.assertNotIn("ringContactSupport", event)


if __name__ == "__main__":
    unittest.main()
