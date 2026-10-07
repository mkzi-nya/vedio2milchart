#!/usr/bin/env python3
"""Small release check that does not depend on a browser or a source chart."""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a generated milplay project.")
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    root = args.project.expanduser().resolve()
    required = [root / "chart.js", root / "audio.m4a", root / "cover.jpg", root / "milplay.zip"]
    missing = [str(p.name) for p in required if not p.is_file()]
    if missing:
        raise SystemExit(f"missing release files: {', '.join(missing)}")
    chart_bytes = (root / "chart.js").stat().st_size
    if chart_bytes > 1_500_000:
        raise SystemExit(f"chart.js is {chart_bytes} bytes, over the 1,500,000 byte limit")
    with zipfile.ZipFile(root / "milplay.zip") as archive:
        bad = archive.testzip()
        names = set(archive.namelist())
        expected = {"chart.js", "audio.m4a", "cover.jpg"}
        if bad:
            raise SystemExit(f"bad ZIP member: {bad}")
        if not expected.issubset(names) or not any(n.startswith("storyboard/") for n in names):
            raise SystemExit(f"incomplete milplay ZIP: {sorted(names)}")
    ffprobe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "json", str(root / "audio.m4a")],
        check=True,
        capture_output=True,
        text=True,
    )
    codec = json.loads(ffprobe.stdout)["streams"][0]["codec_name"]
    if codec != "aac":
        raise SystemExit(f"audio.m4a is encoded as {codec}, expected AAC")
    events_path = root / "events.json"
    events = json.loads(events_path.read_text()) if events_path.exists() else []
    real = [e for e in events if not e.get("isFake")]
    lightning = [e for e in real if int(e.get("type", 0)) == 2]
    bad_events = [e for e in events if float(e.get("duration", 0)) < 0]
    if bad_events:
        raise SystemExit("negative note duration found")
    print(json.dumps({"project": str(root), "chartBytes": chart_bytes, "audioCodec": codec, "notes": len(events), "realNotes": len(real), "lightningNotes": len(lightning), "zipMembers": sorted(names)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
