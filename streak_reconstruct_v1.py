"""Track fast, broad, colored light streaks directly from gameplay frames."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _line_angle(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _runs(bits):
    edges = np.flatnonzero(np.diff(np.r_[False, bits, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def detect_frame(image):
    """Return broad, bright, colored line candidates measured in source pixels."""
    rgb = image.astype(np.float32)
    low = rgb.min(axis=2)
    chroma = rgb.max(axis=2) - low
    bright = (low > 85) & (chroma > 12) & (chroma < 90)
    hough = cv2.HoughLinesP(
        bright.astype(np.uint8) * 255, 1, np.pi / 720,
        threshold=65, minLineLength=300, maxLineGap=65,
    )
    if hough is None:
        return []

    segments = []
    for x1, y1, x2, y2 in hough[:, 0]:
        theta = math.atan2(y2 - y1, x2 - x1) % math.pi
        direction = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-direction[1], direction[0]])
        first = np.array([x1, y1], dtype=float)
        last = np.array([x2, y2], dtype=float)
        segments.append({
            'theta': theta,
            'first': first,
            'last': last,
            'length': float(np.linalg.norm(last - first)),
        })

    # Merge the parallel Hough edges that belong to the same thick stroke.
    groups = []
    for seg in sorted(segments, key=lambda item: -item['length']):
        found = None
        for group in groups:
            angle = _line_angle(seg['theta'], group['theta'])
            d = np.array([math.cos(group['theta']), math.sin(group['theta'])])
            n = np.array([-d[1], d[0]])
            rho = float(((seg['first'] + seg['last']) * .5) @ n)
            if angle < .035 and min(abs(rho - group['rho']), abs(rho + group['rho'])) < 65:
                found = group
                break
        if found is None:
            d = np.array([math.cos(seg['theta']), math.sin(seg['theta'])])
            n = np.array([-d[1], d[0]])
            rho = float(((seg['first'] + seg['last']) * .5) @ n)
            groups.append({'theta': seg['theta'], 'rho': rho, 'segments': [seg]})
        else:
            found['segments'].append(seg)

    candidates = []
    for group in groups:
        segs = group['segments']
        # Average unoriented angles in doubled-angle space so 179° and 1°
        # correctly describe one nearly horizontal stroke.
        weights = np.array([s['length'] for s in segs])
        angles = np.array([s['theta'] for s in segs])
        theta = .5 * math.atan2(float(np.sum(weights * np.sin(2 * angles))),
                               float(np.sum(weights * np.cos(2 * angles))))
        if theta < 0:
            theta += math.pi
        direction = np.array([math.cos(theta), math.sin(theta)])
        normal = np.array([-direction[1], direction[0]])
        midpoints = np.array([(s['first'] + s['last']) * .5 for s in segs])
        rho = float(np.median(midpoints @ normal))
        projected = np.array([[s['first'] @ direction, s['last'] @ direction] for s in segs])
        u0, u1 = float(projected.min()), float(projected.max())
        length = u1 - u0
        if length < 300:
            continue

        offsets = np.arange(-80, 81, 2, dtype=np.float32)
        widths, centers, luminance, color = [], [], [], []
        for u in np.arange(u0, u1 + 1, 4):
            points = direction[None, :] * u + normal[None, :] * (rho + offsets[:, None])
            xx = np.rint(points[:, 0]).astype(int)
            yy = np.rint(points[:, 1]).astype(int)
            in_bounds = (xx >= 0) & (xx < image.shape[1]) & (yy >= 0) & (yy < image.shape[0])
            section = np.zeros(len(offsets), dtype=bool)
            section[in_bounds] = bright[yy[in_bounds], xx[in_bounds]]
            runs = _runs(section)
            if not runs:
                continue
            start, end = max(runs, key=lambda run: run[1] - run[0])
            width = (end - start) * 2
            if width < 18:
                continue
            core = in_bounds[start:end]
            cx, cy = xx[start:end][core], yy[start:end][core]
            if not len(cx):
                continue
            widths.append(width)
            centers.append(float(np.median(offsets[start:end][core])))
            luminance.append(float(np.median(low[cy, cx])))
            color.append(float(np.median(chroma[cy, cx])))

        support = len(widths) / max(1, (u1 - u0) / 4)
        if not widths:
            continue
        width = float(np.median(widths))
        # Thick capsule bodies and rails can also produce long Hough segments;
        # require the full wide, continuous chromatic core of a streak.
        if width < 40 or support < .52:
            continue
        offset = float(np.median(centers))
        center = direction * ((u0 + u1) * .5) + normal * (rho + offset)
        candidates.append({
            'x': float(center[0]), 'y': float(center[1]),
            'rotation': math.degrees(theta), 'length': float(length),
            'width': width, 'support': float(min(1, support)),
            'luminance': float(np.median(luminance)),
            'chroma': float(np.median(color)),
            'alpha': float(np.clip((np.median(luminance) - 60) / 195, .35, 1)),
        })

    # Parallel Hough bands sample the edges of one beam. Keep one representative.
    candidates.sort(key=lambda item: item['length'] * item['width'] * item['support'], reverse=True)
    unique = []
    for candidate in candidates:
        duplicate = False
        for other in unique:
            theta = math.radians(candidate['rotation'])
            other_theta = math.radians(other['rotation'])
            if _line_angle(theta, other_theta) >= .06:
                continue
            direction = np.array([math.cos(theta), math.sin(theta)])
            normal = np.array([-direction[1], direction[0]])
            normal_gap = abs((np.array([candidate['x'] - other['x'], candidate['y'] - other['y']])) @ normal)
            c0, c1 = candidate['x'] * direction[0] + candidate['y'] * direction[1], other['x'] * direction[0] + other['y'] * direction[1]
            gap = max(0, abs(c0 - c1) - (candidate['length'] + other['length']) * .5)
            if normal_gap < 60 and gap < 55:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique[:8]


def _angle_series(samples):
    out = [float(samples[0]['rotation'])]
    for sample in samples[1:]:
        angle = float(sample['rotation'])
        while angle - out[-1] > 90:
            angle -= 180
        while angle - out[-1] < -90:
            angle += 180
        out.append(angle)
    return out


def _track_quality(samples):
    if len(samples) < 2:
        return False
    widths = np.array([s['width'] for s in samples])
    lengths = np.array([s['length'] for s in samples])
    support = np.array([s['support'] for s in samples])
    if np.median(widths) < 46 or np.median(lengths) < 550 or np.median(support) < .84:
        return False
    # A rendered hold is itself a long, brightly colored capsule. Its width to
    # length ratio is larger than a screen-spanning beam, even when perspective
    # or Hough grouping makes it look like a line.
    aspect = float(np.median(widths / lengths))
    if aspect > .08:
        return False
    angles = _angle_series(samples)
    angular_span = max(angles) - min(angles)
    angle_rates = [abs(angles[i] - angles[i - 1]) / max(1e-4, samples[i]['time'] - samples[i - 1]['time'])
                   for i in range(1, len(samples))]
    center_rates = [math.hypot(samples[i]['x'] - samples[i - 1]['x'],
                               samples[i]['y'] - samples[i - 1]['y']) /
                    max(1e-4, samples[i]['time'] - samples[i - 1]['time'])
                    for i in range(1, len(samples))]
    # Ordinary capsule tracks remain upright and move at note speed. A genuine
    # light streak either changes its axis substantially or sweeps across the
    # screen at an unmistakably higher rate.
    if angular_span >= 25:
        return True
    median_angle = float(np.median(angles))
    vertical = _line_angle(math.radians(median_angle), math.pi / 2) < .28
    max_angle_rate = max(angle_rates, default=0)
    max_center_rate = max(center_rates, default=0)
    if max_center_rate >= 3000 and aspect < .07:
        return True
    if vertical:
        return max_angle_rate >= 250 and aspect < .055
    return max_angle_rate >= 100 or (max_center_rate >= 1200 and aspect < .065)


def _append_streak(tracks, samples):
    if len(samples) < 2:
        return
    angles = _angle_series(samples)
    for sample, angle in zip(samples, angles):
        sample['rotation'] = float(angle)
    tracks.append({'kind': 'streak', 'samples': samples})


def _assemble_tracks(frames, fps):
    tracks, active = [], []
    for frame, candidates in enumerate(frames):
        time = frame / fps
        active = [tr for tr in active if frame - tr['_frame'] <= 2]
        used = set()
        if active and candidates:
            costs = np.full((len(active), len(candidates)), 1e6, dtype=float)
            for i, track in enumerate(active):
                samples = track['samples']
                previous = samples[-1]
                dt = time - previous['time']
                predicted_x, predicted_y = previous['x'], previous['y']
                predicted_angle = previous['rotation']
                if len(samples) >= 2:
                    old = samples[-2]
                    old_dt = max(1 / fps, previous['time'] - old['time'])
                    predicted_x += (previous['x'] - old['x']) / old_dt * dt
                    predicted_y += (previous['y'] - old['y']) / old_dt * dt
                    da = (previous['rotation'] - old['rotation'] + 90) % 180 - 90
                    predicted_angle += da / old_dt * dt
                for j, candidate in enumerate(candidates):
                    angle = math.radians(candidate['rotation'])
                    angle_diff = _line_angle(angle, math.radians(predicted_angle))
                    distance = math.hypot(candidate['x'] - predicted_x, candidate['y'] - predicted_y)
                    width_diff = abs(math.log(candidate['width'] / previous['width']))
                    if angle_diff < .65 and distance < 460 and width_diff < .6:
                        costs[i, j] = distance + angle_diff * 600 + width_diff * 120
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                track = active[i]
                track['samples'].append({'time': time, **candidates[j]})
                track['_frame'] = frame
                used.add(j)
        for j, candidate in enumerate(candidates):
            if j in used:
                continue
            track = {'kind': 'streak', 'samples': [{'time': time, **candidate}], '_frame': frame}
            tracks.append(track)
            active.append(track)

    output = []
    for track in tracks:
        samples = track['samples']
        if not _track_quality(samples):
            continue
        # Keep only confirmed moving effects. Gaps of at most two frames are
        # interpolated by the renderer between the surrounding measurements.
        output.append({'kind': 'streak', 'samples': samples})
    return output


def write_sprite(path):
    """Create a compact, aspect-independent pale cyan light-streak texture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1024, 128
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    tip = np.clip((1 - np.abs(x)) / .12, 0, 1)
    tip = tip * tip * (3 - 2 * tip)
    halo = np.exp(-.5 * (y / .42) ** 2)
    core = np.exp(-.5 * (y / .14) ** 2)
    alpha = np.clip(tip * (.70 * halo + .44 * core), 0, 1)
    blue = np.full((h, w), 255, dtype=np.float32)
    green = 195 + 60 * core
    red = 145 + 110 * core
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.broadcast_to(red, (h, w)).astype(np.uint8)
    rgba[..., 1] = np.broadcast_to(green, (h, w)).astype(np.uint8)
    rgba[..., 2] = np.broadcast_to(blue, (h, w)).astype(np.uint8)
    rgba[..., 3] = np.broadcast_to(alpha * 255, (h, w)).astype(np.uint8)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def scan(video, root):
    root = Path(root)
    cache = root / 'streaks-v1.json'
    stat = Path(video).stat()
    signature = hashlib.sha256(
        Path(__file__).read_bytes() + f'{stat.st_size}:{stat.st_mtime_ns}'.encode()
    ).hexdigest()
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get('signature') == signature:
            tracks = saved['tracks']
            if tracks:
                write_sprite(root / 'storyboard' / 'light-streak.png')
            return tracks, saved['report']

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    candidates_by_frame = []
    frame = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        candidates_by_frame.append(detect_frame(image))
        frame += 1
        if frame % 600 == 0:
            print('V1 broad-streak scan', frame, 'frames', flush=True)
    cap.release()

    tracks = _assemble_tracks(candidates_by_frame, fps)
    report = {
        'frames': frame,
        'fps': float(fps),
        'tracks': len(tracks),
        'samples': sum(len(track['samples']) for track in tracks),
        'method': 'full-frame chromatic ridge width plus fast axis rotation/translation',
    }
    cache.write_text(json.dumps({'signature': signature, 'tracks': tracks, 'report': report}, separators=(',', ':')))
    (root / 'streak-review-v1.json').write_text(json.dumps(report, indent=2))
    if tracks:
        write_sprite(root / 'storyboard' / 'light-streak.png')
    return tracks, report
