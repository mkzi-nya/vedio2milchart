import unittest

from judgement_fit_v4 import _counter_transition_windows


class CounterTransitionWindowTests(unittest.TestCase):
    def test_only_observed_combo_rises_create_windows(self):
        rows = [
            {"time": 1.0, "combo": 5, "error": .01, "margin": .08},
            {"time": 1.1, "combo": 5, "error": .01, "margin": .08},
            {"time": 1.2, "combo": 6, "error": .01, "margin": .08},
        ]
        window = _counter_transition_windows(rows)[0]
        self.assertEqual(window[:2], (1.1, 1.2))
        self.assertAlmostEqual(window[2], .08)
        self.assertEqual(window[3], 1)

    def test_uncertain_counter_reads_do_not_create_transition(self):
        rows = [
            {"time": 2.0, "combo": 5, "error": .2, "margin": .001},
            {"time": 2.033, "combo": 6, "error": .2, "margin": .001},
        ]
        self.assertEqual(_counter_transition_windows(rows), [])

    def test_gap_widens_tolerance_but_is_capped(self):
        rows = [
            {"time": 3.0, "combo": 5, "error": .01, "margin": .08},
            {"time": 3.5, "combo": 6, "error": .01, "margin": .08},
        ]
        self.assertEqual(_counter_transition_windows(rows), [(3.0, 3.5, .16, 1)])


if __name__ == "__main__":
    unittest.main()
