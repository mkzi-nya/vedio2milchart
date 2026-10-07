import unittest

from hollow_notes_v6 import restore_unrepresented_visual_tracks


def track(identifier, y0=100):
    return {
        'id': identifier,
        'speed': 960,
        'samples': [[i / 30, 372, y0 + i * 32, 90, 52, 88]
                    for i in range(12)],
    }


class HollowVisualRecoveryTests(unittest.TestCase):
    def test_restores_only_unrepresented_sustained_tracks_as_non_scoring(self):
        existing = {
            'time': .5, 'duration': 0, 'isFake': True, 'type': 1,
            'path': [[i / 30, 372, 100 + i * 32] for i in range(12)],
        }
        events = [existing]
        summary = restore_unrepresented_visual_tracks(
            events, [track(10), track(11, 138)])

        self.assertEqual(summary['restoredVisualTracks'], 1)
        recovered = next(e for e in events if e.get('hollowTrack') == 11)
        self.assertTrue(recovered['isFake'])
        self.assertTrue(recovered['visualOnly'])
        self.assertEqual(recovered['type'], 1)
        self.assertEqual(recovered['duration'], 0)
        self.assertEqual(len(recovered['path']), 12)

    def test_short_or_low_shape_confidence_tracks_are_not_added(self):
        short = track(20)
        short['samples'] = short['samples'][:5]
        weak = track(21)
        for sample in weak['samples']:
            sample[5] = 110
        events = []

        summary = restore_unrepresented_visual_tracks(events, [short, weak])

        self.assertEqual(summary['restoredVisualTracks'], 0)
        self.assertEqual(events, [])

    def test_repeated_recovery_does_not_duplicate_its_own_output(self):
        events = []
        tracks = [track(30)]

        first = restore_unrepresented_visual_tracks(events, tracks)
        second = restore_unrepresented_visual_tracks(events, tracks)

        self.assertEqual(first['restoredVisualTracks'], 1)
        self.assertEqual(second['restoredVisualTracks'], 0)
        self.assertEqual(len(events), 1)


if __name__ == '__main__':
    unittest.main()
