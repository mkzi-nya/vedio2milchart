import unittest

from visual_dedupe import collapse_duplicate_visual_tracks


def event(track, *, fake=True, y=100, count=12, note_type=0):
    path=[[i / 30, 500 - i * 20, y + i * 20] for i in range(count)]
    return {'time':path[-1][0], 'duration':0, 'type':note_type,
            'isFake':fake, 'frames':count, 'confidence':'medium',
            'evidence':'measured-path', 'path':path}


class VisualDedupeTests(unittest.TestCase):
    def test_collapses_overlapping_fake_detection_but_keeps_longer_track(self):
        short=event(1,count=8)
        long=event(2,count=15)
        events=[short,long]

        result=collapse_duplicate_visual_tracks(events)

        self.assertEqual(result['removedDuplicateVisualTracks'],1)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['frames'],15)

    def test_nearby_parallel_tracks_remain_distinct(self):
        events=[event(1,y=100),event(2,y=138)]

        result=collapse_duplicate_visual_tracks(events)

        self.assertEqual(result['removedDuplicateVisualTracks'],0)
        self.assertEqual(len(events),2)

    def test_two_scored_notes_are_never_deduplicated(self):
        events=[event(1,fake=False),event(2,fake=False)]

        result=collapse_duplicate_visual_tracks(events)

        self.assertEqual(result['removedDuplicateVisualTracks'],0)
        self.assertEqual(len(events),2)


if __name__ == '__main__':
    unittest.main()
