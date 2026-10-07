import unittest

import numpy as np

from special_flow_reconcile import collapse_duplicate_hold_endpoint_passes


def sampled_path(start, end, offset=0.0):
    out = []
    for t in np.linspace(start, end, 10):
        out.append([float(t), float(300 + 25 * t + offset), float(200 + 40 * t)])
    return out


class SpecialFlowReconcileTests(unittest.TestCase):
    def make_pair(self, *, secondary_offset=2.0, source_contact=1.0):
        first = {
            "time": 1.0, "duration": .40, "speed": 1260, "type": 0,
            "track": 42, "sourceContactTime": source_contact,
            "ringHeadSupport": .85, "ringContactSupport": .90,
            "isFake": False, "path": sampled_path(.70, 1.37),
            "tailPath": sampled_path(.70, 1.37, offset=-80),
        }
        duplicate = {
            "time": 1.12, "duration": .28, "speed": 1260, "type": 0,
            "track": 42, "sourceContactTime": source_contact,
            "ringHeadSupport": .30, "ringContactSupport": .70,
            "isFake": False, "path": sampled_path(.70, 1.37, offset=secondary_offset),
            "tailPath": sampled_path(.70, 1.37, offset=-80 + secondary_offset),
        }
        return first, duplicate

    def test_removes_weaker_same_capsule_pass_when_both_endpoints_overlap(self):
        first, duplicate = self.make_pair()
        events = [first, duplicate]

        result = collapse_duplicate_hold_endpoint_passes(events)

        self.assertEqual(result["removedDuplicateEndpointPasses"], 1)
        self.assertEqual(events, [first])
        self.assertLess(result["findings"][0]["headPathMedianPx"], 3)
        self.assertLess(result["findings"][0]["tailPathMedianPx"], 3)

    def test_keeps_a_second_head_when_it_has_a_distinct_source_contact(self):
        first, duplicate = self.make_pair()
        first["sourceContactTime"] = 1.0
        duplicate["sourceContactTime"] = 1.12
        events = [first, duplicate]

        result = collapse_duplicate_hold_endpoint_passes(events)

        self.assertEqual(result["removedDuplicateEndpointPasses"], 0)
        self.assertEqual(len(events), 2)

    def test_keeps_separate_capsule_trajectories(self):
        first, duplicate = self.make_pair(secondary_offset=40)
        events = [first, duplicate]

        result = collapse_duplicate_hold_endpoint_passes(events)

        self.assertEqual(result["removedDuplicateEndpointPasses"], 0)
        self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
