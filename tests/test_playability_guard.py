import unittest

from playability_guard import nudge_early_contacts


class PlayabilityGuardTests(unittest.TestCase):
    def test_one_source_frame_nudge_preserves_hold_end(self):
        events = [
            {"time": 1.0, "duration": .4, "isFake": False, "type": 0,
             "ringContactSupport": .8, "ringHeadSupport": .7},
        ]

        changes = nudge_early_contacts(events, [{"id": 0, "t": 1.0, "judged": .85}], 30)

        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(events[0]["time"], 1 + 1/30, places=5)
        self.assertAlmostEqual(events[0]["time"] + events[0]["duration"], 1.4, places=5)

    def test_very_strong_measured_contact_is_not_moved(self):
        events = [{"time": 1.0, "duration": 0, "isFake": False, "type": 1,
                   "sourceContactTime": 1.0, "ringContactSupport": .95,
                   "ringHeadSupport": .93}]

        changes = nudge_early_contacts(events, [{"id": 0, "t": 1.0, "judged": .84}], 30)

        self.assertEqual(changes, [])
        self.assertEqual(events[0]["time"], 1.0)


if __name__ == "__main__":
    unittest.main()
