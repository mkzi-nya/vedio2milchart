import unittest

import cv2
import numpy as np

from streak_reconstruct_v1 import _assemble_tracks, detect_frame


class BroadStreakTests(unittest.TestCase):
    def test_detects_a_long_chromatic_beam(self):
        frame = np.full((720, 1280, 3), (18, 13, 11), dtype=np.uint8)
        cv2.line(frame, (90, 640), (1120, 95), (255, 232, 224), 52, cv2.LINE_AA)
        candidates = detect_frame(frame)
        self.assertTrue(candidates)
        beam = max(candidates, key=lambda q: q['length'] * q['width'])
        self.assertGreater(beam['length'], 800)
        self.assertGreaterEqual(beam['width'], 44)
        self.assertGreater(beam['support'], .8)

    def test_rejects_a_steady_falling_capsule_track(self):
        frames = []
        for i in range(5):
            frames.append([{
                'x': 500., 'y': 200. + i * 14,
                'rotation': 90., 'length': 700., 'width': 52.,
                'support': .95, 'luminance': 220., 'chroma': 25., 'alpha': .82,
            }])
        self.assertEqual(_assemble_tracks(frames, 30), [])

    def test_keeps_a_fast_sweeping_streak_track(self):
        frames = []
        for i in range(4):
            frames.append([{
                'x': 430. - i * 30, 'y': 500. - i * 140,
                'rotation': 150. + i * 12, 'length': 900. - i * 80,
                'width': 50., 'support': .92, 'luminance': 220.,
                'chroma': 28., 'alpha': .82,
            }])
        tracks = _assemble_tracks(frames, 30)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]['samples']), 4)
        self.assertGreater(tracks[0]['samples'][-1]['rotation'], 180)


if __name__ == '__main__':
    unittest.main()
