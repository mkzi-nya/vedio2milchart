import json
import unittest

from export_v2 import export_js


class StreakExportTests(unittest.TestCase):
    def test_rotation_sign_and_180_degree_seam_are_continuous(self):
        samples = [
            {'time': 1.0, 'x': 600, 'y': 300, 'rotation': 154.7,
             'length': 900, 'width': 52, 'alpha': .8},
            {'time': 1.033333, 'x': 550, 'y': 250, 'rotation': 168.8,
             'length': 800, 'width': 52, 'alpha': .8},
            {'time': 1.066666, 'x': 530, 'y': 150, 'rotation': .456,
             'length': 780, 'width': 52, 'alpha': .8},
            {'time': 1.099999, 'x': 540, 'y': 75, 'rotation': 10.6,
             'length': 650, 'width': 50, 'alpha': .8},
        ]
        source = export_js([], effects=[{'kind': 'streak', 'samples': samples}], duration=2)
        rotation_call = next(line for line in source.splitlines()
                             if 'c(sbx[0],2,4,' in line)
        packed = rotation_call.split('c(sbx[0],2,4,', 1)[1].split(');', 1)[0]
        rotations = json.loads(packed)

        self.assertEqual(len(rotations), 4)
        self.assertAlmostEqual(rotations[0][1], -154.7)
        self.assertAlmostEqual(rotations[-1][1], -190.6)
        self.assertTrue(all(abs(b[1] - a[1]) < 20
                            for a, b in zip(rotations, rotations[1:])))

    def test_storyboard_height_matches_measured_streak_core_width(self):
        samples = [
            {'time': 1.0, 'x': 600, 'y': 300, 'rotation': 140,
             'length': 800, 'width': 52, 'alpha': .8},
            {'time': 1.033333, 'x': 550, 'y': 250, 'rotation': 145,
             'length': 800, 'width': 52, 'alpha': .8},
        ]
        source = export_js([], effects=[{'kind': 'streak', 'samples': samples}], duration=2)
        height_call = next(line for line in source.splitlines()
                           if 'c(sbx[0],2,11,' in line)
        packed = height_call.split('c(sbx[0],2,11,', 1)[1].split(');', 1)[0]
        heights = json.loads(packed)

        self.assertEqual(len(heights), 2)
        self.assertAlmostEqual(heights[0][1], 52 * 8 / (800 * .83) * 2.3, places=3)

    def test_fake_hold_duration_reaches_observed_visual_endpoint(self):
        event = {
            'time': 1.0,
            'duration': .5,
            'type': 0,
            'isFake': True,
            'x': 640,
            'judgeY': .82,
            'rotation': 90,
            'path': [[1.0, 640, 530], [1.4, 640, 300], [1.8, 640, 25]],
        }
        source = export_js([event], duration=3)

        note_call = next(line for line in source.splitlines() if 'm.note(' in line)
        self.assertIn(',1.00000,1.83333,0,true,', note_call)

    def test_fake_non_hold_stays_drawable_through_observed_path(self):
        event = {
            'time': 1.0,
            'duration': 0,
            'type': 1,
            'isFake': True,
            'x': 372,
            'judgeY': .82,
            'rotation': 90,
            'path': [[1.0, 372, 520], [1.4, 372, 300], [1.8, 372, 25]],
            'opacityPath': [[1.0, .75], [1.4, .75], [1.8, .65]],
        }
        source = export_js([event], duration=3)

        note_call = next(line for line in source.splitlines() if 'm.note(' in line)
        self.assertIn(',1.83333,1.83333,1,true,', note_call)
        self.assertIn('a(n,1,2,', source)


if __name__ == '__main__':
    unittest.main()
