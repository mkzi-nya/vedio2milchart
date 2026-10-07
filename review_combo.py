"""Independent source-crop OCR for an exact-frame review, with native runtime counts."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def review(root, folder):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    rows = json.loads((folder / 'metrics.json').read_text())
    jobs, lookup = {}, {}
    for row in rows:
        im = cv2.imread(str(folder / f"source-{row['time']}.jpg"))[62:126, 510:770].min(2)
        mask = np.zeros_like(im)
        mask[:, (im > 150).sum(0) > 48] = 255
        im = cv2.inpaint(im, mask, 3, cv2.INPAINT_TELEA)
        key = hashlib.sha256(im.tobytes()).hexdigest()
        lookup[row['frame']] = key
        jobs[key] = cv2.imencode('.png', cv2.resize(im, (1040, 256)))[1].tobytes()

    def read(pair):
        key, png = pair
        proc = subprocess.run(['tesseract', 'stdin', 'stdout', '--psm', '7', '-c',
                               'tessedit_char_whitelist=0123456789', 'tsv'],
                              input=png, capture_output=True, check=True)
        words, confs = [], []
        for line in proc.stdout.decode().splitlines()[1:]:
            q = line.split('\t')
            if len(q) >= 12 and q[11].isdigit():
                words.append(q[11]); confs.append(float(q[10]))
        return key, (int(''.join(words)) if words else None, min(confs) if confs else 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(read, jobs.items()))
    for row in rows:
        row['videoCombo'], row['ocrConfidence'] = results[lookup[row['frame']]]
    data = folder / 'combo-review.json'
    data.write_text(json.dumps(rows, indent=2))
    js = '''const fs=require('node:fs');const {createHarness}=require(process.argv[1]);
    (async()=>{const h=createHarness(process.argv[2]);h.context.src=fs.readFileSync(process.argv[3],'utf8');
    const p=await h.run('parseText(src,"chart.js")');h.context.chart=p.chart;
    h.context.times=JSON.parse(fs.readFileSync(process.argv[4])).map(q=>q.time);
    const counts=h.run('(()=>{const rt=makeRuntime(chart);return times.map(t=>rt.comboAt(t))})()');
    const rows=JSON.parse(fs.readFileSync(process.argv[4]));rows.forEach((q,i)=>{q.renderCombo=counts[i];q.difference=q.videoCombo===null?null:q.videoCombo-counts[i]});
    fs.writeFileSync(process.argv[4],JSON.stringify(rows,null,2));})().catch(e=>{console.error(e);process.exitCode=1});'''
    subprocess.run(['node', '-e', js, str(HERE.parent / 'milplay/tests/harness'),
                    str(HERE.parent / 'milplay'), str(root / 'chart.js'), str(data)], check=True)
    print(data)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('folder')
    a = p.parse_args(); review(a.root, a.folder)
