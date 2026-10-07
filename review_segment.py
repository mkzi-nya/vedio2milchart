"""Compare exact video-frame timestamps with the native milplay renderer."""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def review(root, start, end, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(root / 'source.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = range(round(start * fps), round(end * fps) + 1)
    times = [round(f / fps, 8) for f in frames]
    subprocess.run(['node', str(HERE / 'render_review.cjs'), str(root),
                    ','.join(map(str, times)), str(out), '1280x720'], check=True)
    sheets, metrics = [], []
    for i, (f, t) in enumerate(zip(frames, times)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read frame {f}')
        a = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        p = out / f'milplay-{t}.png'
        if not p.exists():
            p = out / f'milplay-{t:g}.png'
        b = np.array(Image.open(p).convert('RGB'))
        diff = np.abs(a.astype(float) - b.astype(float))
        diff[:112] = 0
        d = np.clip(diff * 3, 0, 255).astype('uint8')
        metrics.append({'frame': f, 'time': t, 'meanAbsRGBBelowHUD': float(diff[112:].mean())})
        Image.fromarray(a).save(out / f'source-{t}.jpg', quality=95)
        if i % 6 == 0:
            sheets.append(Image.new('RGB', (1440, 1764), (20, 20, 20)))
        sheet = sheets[-1]
        draw = ImageDraw.Draw(sheet)
        y = (i % 6) * 294
        draw.text((5, y + 5), f'{f} / {t:.8f}s  SOURCE | RENDER | DIFF x3', fill='white')
        for j, pix in enumerate([a, b, d]):
            sheet.paste(Image.fromarray(pix).resize((480, 270)), (480 * j, y + 24))
    for i, sheet in enumerate(sheets):
        sheet.save(out / f'sheet-{i+1:03}.jpg', quality=94)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    cap.release()
    print(f'Compared every frame: {len(times)} at source FPS {fps}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('start', type=float)
    p.add_argument('end', type=float)
    p.add_argument('out')
    a = p.parse_args()
    review(a.root, a.start, a.end, a.out)
