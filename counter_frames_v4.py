"""Read every video counter frame with digit templates learned from this video."""
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np


def template_errors(totals, hits, observed_pixels):
    """Score template matches without unsigned subtraction underflow."""
    total_pixels = np.asarray(totals, dtype=np.float64)
    hit_pixels = np.asarray(hits, dtype=np.float64)
    observed_pixels = int(observed_pixels)
    excess = np.maximum(0.0, observed_pixels-hit_pixels)
    return (total_pixels-hit_pixels+excess*.6) / np.maximum(
        1.0, total_pixels+observed_pixels*.6)


def scan(video, root):
    root = Path(root)
    final_combo=int(json.loads((root/'combo.json').read_text())['finalCombo'])
    labels = {round(q['time']*30): str(q['combo']) for q in json.loads((root/'counter-probes.json').read_text())
              if q['combo'] is not None and q['confidence'] >= 95 and 0 <= q['combo'] <= final_combo}
    cap = cv2.VideoCapture(str(video)); frames, records = [], []
    bank = {str(i): [] for i in range(10)}
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        crop = im[70:104, 535:745]
        lo = crop.min(2)
        mask = ((lo > 190) & (crop.max(2)-lo < 36)).astype(np.uint8)
        # Full-height neutral trails are occluders, including those crossing a digit.
        ignore = (lo > 145).sum(0) > 29
        frames.append((mask, ignore))
        if f in labels and not ignore.any():
            cols = np.flatnonzero(mask.sum(0) > 0)
            groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1)+1)
            glyphs = []
            for g in groups:
                if len(g) < 7:
                    continue
                ys = np.flatnonzero(mask[:, g].sum(1))
                if not 20 <= len(ys) <= 29:
                    continue
                glyphs.append((g[0], g[-1], ys[0], ys[-1]))
            label = labels[f]
            if len(glyphs) == len(label):
                records.append((f, label, glyphs))
                for d, (left, right, top, bottom) in zip(label, glyphs):
                    bank[d].append(mask[top:bottom+1, left:right+1])
        f += 1
    cap.release()
    prototypes = {}
    for d, samples in bank.items():
        if not samples:
            raise ValueError('Video template missing digit '+d)
        shape = Counter(q.shape for q in samples).most_common(1)[0][0]
        prototypes[d] = (np.median(np.stack([q for q in samples if q.shape == shape]), axis=0) >= .5).astype(np.uint8)
    # Estimate source font advance and per-glyph bearings from labelled frames.
    design, observed = [], []
    for _, label, glyphs in records:
        for j, (d, glyph) in enumerate(zip(label, glyphs)):
            row = [j-(len(label)-1)/2]+[float(d == str(i)) for i in range(10)]
            design.append(row); observed.append((glyph[0]+glyph[1])/2)
    fit = np.linalg.lstsq(np.asarray(design), observed, rcond=None)[0]
    templates = []
    for value in range(final_combo+1):
        mask = np.zeros((34, 210), np.uint8)
        label = str(value)
        for j, d in enumerate(label):
            glyph = prototypes[d]; h, w = glyph.shape
            center = fit[0]*(j-(len(label)-1)/2)+fit[int(d)+1]
            left = int(round(center-(w-1)/2))
            mask[5:5+h, left:left+w] = glyph
        templates.append(mask)
    templates = np.stack(templates)
    # Clean source OCR frames provide range bounds, never a synthetic count.
    anchors = [(0, 0)]
    for frame, label, _ in records:
        c = int(label); t = frame/30
        if c >= anchors[-1][1] and c-anchors[-1][1] <= (t-anchors[-1][0])*42+8:
            anchors.append((t, c))
    anchors.append((len(frames)/30, final_combo))
    times = np.asarray([q[0] for q in anchors]); counts = np.asarray([q[1] for q in anchors])
    results = []
    for frame, (observed, ignore) in enumerate(frames):
        t = frame/30; j = int(np.searchsorted(times, t, side='right'))
        left = max(0, int(counts[max(0, j-1)])-3)
        right = min(final_combo, int(counts[min(j, len(counts)-1)])+3)
        if right < left:
            continue
        # One pixel tolerance absorbs MPEG ringing and fractional text bearings.
        observed[:, ignore] = 0
        valid = (~ignore)[None, :]
        expanded = cv2.dilate(observed, np.ones((3, 3), np.uint8))
        predicted = templates[left:right+1]
        totals = np.sum(predicted*valid, axis=(1, 2))
        hits = np.sum(predicted*expanded*valid, axis=(1, 2))
        # Reductions here use unsigned integer arrays. The helper converts
        # before subtraction so a dilated glyph match cannot wrap around.
        errors = template_errors(totals, hits, observed.sum())
        k = int(np.argmin(errors)); error = float(errors[k]); confidence = 1-error
        # An ambiguous partially hidden count is omitted, not interpolated.
        sorted_errors = np.sort(errors)
        margin = float(sorted_errors[1]-error) if len(errors)>1 else 1.
        if observed.sum() >= 35 and error < .16 and margin > .018:
            results.append({'time': round(t, 5), 'combo': left+k, 'error': round(error, 5), 'margin': round(margin, 5)})
    # Demand persistence on two adjacent source frames; discard implausible jumps.
    confirmed = []; last = (0., 0)
    for a, b in zip(results, results[1:]):
        if b['time']-a['time'] > .035 or a['combo'] != b['combo']:
            continue
        if a['combo'] < last[1] or a['combo']-last[1] > (a['time']-last[0])*45+7:
            continue
        confirmed.append(a); last = (a['time'], a['combo'])
    report = {'frames': len(frames), 'readableFrames': len(results), 'confirmedFrames': len(confirmed),
              'digitTemplates': {d: list(p.shape) for d, p in prototypes.items()},
              'fontAdvancePixels': round(float(fit[0]), 5), 'finalCombo': last[1]}
    (root/'counter-frames-v4.json').write_text(json.dumps({'samples': confirmed, 'report': report}, indent=2))
    print('Per-frame video counter:', report, flush=True)
    return confirmed, report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('video'); p.add_argument('--output', required=True)
    a = p.parse_args(); scan(a.video, a.output)
