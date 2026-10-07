import unittest

from counter_guard import guard_fit, pin_drifted_contacts


def sample(time, combo):
    return {"time": time, "combo": combo, "error": 0.0, "margin": 0.1}


class CounterGuardTests(unittest.TestCase):
    def test_regressing_refit_restores_seed_time_and_combo(self):
        seed = [{"time": 1.0, "duration": 0.0, "isFake": False},
                {"time": 2.0, "duration": 0.0, "isFake": False}]
        fitted = [{**seed[0], "time": 1.5}, dict(seed[1])]
        observations = [sample(0.0, 0), sample(1.0, 1), sample(1.5, 1),
                        sample(2.0, 2), sample(3.0, 2)]

        result, report = guard_fit(seed, fitted, observations, final_combo=2)

        self.assertEqual(report["state"], "conservative-counter-merge")
        self.assertEqual(result[0]["time"], 1.0)
        self.assertEqual(report["resultJudgements"], 2)
        self.assertLess(report["resultWeightedLoss"], report["fitWeightedLoss"])

    def test_improving_refit_is_kept(self):
        seed = [{"time": 1.5, "duration": 0.0, "isFake": False},
                {"time": 2.0, "duration": 0.0, "isFake": False}]
        fitted = [{**seed[0], "time": 1.0}, dict(seed[1])]
        observations = [sample(0.0, 0), sample(1.0, 1), sample(1.5, 1),
                        sample(2.0, 2), sample(3.0, 2)]

        result, report = guard_fit(seed, fitted, observations, final_combo=2)

        self.assertEqual(report["state"], "kept-improving-fit")
        self.assertEqual(result[0]["time"], 1.0)

    def test_strong_measured_contact_is_not_overruled_by_counter_fit(self):
        seed = [{"time": 1.0, "duration": 0.0, "isFake": False,
                 "sourceContactTime": 1.0, "ringContactSupport": .95,
                 "ringHeadSupport": .9}]
        fitted = [{**seed[0], "time": 1.1}]
        observations = [sample(0.0, 0), sample(1.0, 0), sample(1.1, 1),
                        sample(2.0, 1)]

        result, report = guard_fit(seed, fitted, observations, final_combo=1)

        self.assertEqual(report["state"], "conservative-counter-merge")
        self.assertEqual(result[0]["time"], 1.0)
        self.assertEqual(report["resultJudgements"], 1)

    def test_rechecks_newly_drifted_contacts_after_a_refit(self):
        events = [
            {"time": 1.0, "isFake": False, "sourceContactTime": 1.06,
             "ringContactSupport": .95, "ringHeadSupport": .9},
            {"time": 2.0, "isFake": False, "sourceContactTime": 2.06,
             "ringContactSupport": .95, "ringHeadSupport": .9,
             "counterPinnedContact": True},
            {"time": 3.0, "isFake": True, "sourceContactTime": 3.06,
             "ringContactSupport": .95, "ringHeadSupport": .9},
            {"time": 4.0, "isFake": False, "sourceContactTime": 4.06,
             "ringContactSupport": .4, "ringHeadSupport": .9},
            {"time": 5.0, "isFake": False, "sourceContactTime": 5.2,
             "ringContactSupport": .95, "ringHeadSupport": .9},
        ]

        pinned = pin_drifted_contacts(events)

        self.assertEqual(pinned, [0])
        self.assertTrue(events[0]["counterPinnedContact"])
        self.assertTrue(events[1]["counterPinnedContact"])


if __name__ == "__main__":
    unittest.main()
