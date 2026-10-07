#!/usr/bin/env python3
"""Convert every supplied gameplay recording and publish verified ZIPs."""
from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path.home() / "Downloads"
PYTHON = Path(sys.executable)


def slug(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", name).strip("-")
    return value or "video"


def convert(video: Path, force: bool = False) -> dict:
    output = DOWNLOADS / "milthm-video-outputs" / slug(video.stem)
    output.mkdir(parents=True, exist_ok=True)
    if not force and (output / "milplay.zip").is_file():
        check = subprocess.run([str(PYTHON), str(ROOT / "verify_project.py"), str(output)], cwd=ROOT, capture_output=True, text=True)
        if check.returncode == 0:
            report = json.loads(check.stdout)
            published = DOWNLOADS / f"{slug(video.stem)}.zip"
            if not published.exists() or published.stat().st_mtime < (output / "milplay.zip").stat().st_mtime:
                shutil.copyfile(output / "milplay.zip", published)
            return {"video": str(video), "output": str(output), "zip": str(published), "reused": True, **report}
    log = output / "conversion.log"
    command = [str(PYTHON), str(ROOT / "convert.py"), str(video), "--output", str(output), "--title", video.stem, "--resume"]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"conversion failed for {video.name}; see {log}")
    check = subprocess.run([str(PYTHON), str(ROOT / "verify_project.py"), str(output)], cwd=ROOT, check=True, capture_output=True, text=True)
    report = json.loads(check.stdout)
    published = DOWNLOADS / f"{slug(video.stem)}.zip"
    shutil.copyfile(output / "milplay.zip", published)
    return {"video": str(video), "output": str(output), "zip": str(published), **report}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Convert all video_*.mp4 recordings in Downloads.")
    parser.add_argument("--force", action="store_true", help="rebuild even when a verified output already exists")
    args = parser.parse_args()
    videos = sorted(DOWNLOADS.glob("video_*.mp4"))
    if not videos:
        raise SystemExit("no video_*.mp4 files found in Downloads")
    results = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(convert, video, args.force): video for video in videos}
        for future in concurrent.futures.as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - release runner diagnostics
                failures.append({"video": str(video), "error": str(exc)})
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
            else:
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    manifest = DOWNLOADS / "milthm-video-outputs" / "batch-report.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"results": sorted(results, key=lambda q: q["video"]), "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
