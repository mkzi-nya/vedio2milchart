#!/usr/bin/env python3
"""Public command-line entry point for video-to-milplay conversion."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a complete Milthm gameplay recording into a playable milplay ZIP."
    )
    parser.add_argument("video", type=Path, help="complete gameplay recording")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--title", default=None, help="chart title (defaults to the video filename)")
    parser.add_argument("--resume", action="store_true", help="reuse verified pixel-scan caches")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        parser.error(f"video not found: {video}")
    output = args.output.expanduser().resolve()
    title = args.title or video.stem
    command = [sys.executable, str(ROOT / "automatic.py"), str(video), "--output", str(output), "--title", title]
    if args.resume:
        command.append("--resume")
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
