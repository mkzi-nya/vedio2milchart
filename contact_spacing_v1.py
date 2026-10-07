"""Separate very close, visually supported tap/hold contacts by one frame.

Some source charts place a weak tap immediately before a strongly supported
hold head on the same moving judgement line.  In Milplay's touch resolver the
early tap can fall inside the hold head's acceptance window and steal that
touch, leaving the tap unjudged.  Preserve the measured event locations and
hold release time while opening a one-frame gap around that contact pair.
"""
from __future__ import annotations

import math


def adjust(events, fps=30.0):
    frame = 1.0 / max(1.0, float(fps))
    notes = sorted((e for e in events if not e.get("isFake")), key=lambda e: float(e["time"]))
    used = set()
    changes = []
    for i, tap in enumerate(notes):
        if id(tap) in used or float(tap.get("duration", 0)) > .015:
            continue
        if int(tap.get("type", 0)) == 2 or int(tap.get("frames", 0)) > 4:
            continue
        tap_ring = float(tap.get("ringContactSupport", 0) or 0)
        tap_head = float(tap.get("ringHeadSupport", 0) or 0)
        if not (.50 <= tap_ring < .80 and tap_head < .20):
            continue
        for hold in notes[i + 1:]:
            gap = float(hold["time"]) - float(tap["time"])
            if gap > .12:
                break
            duration = float(hold.get("duration", 0) or 0)
            if duration < .18 or int(hold.get("type", 0)) == 2:
                continue
            hold_ring = float(hold.get("ringContactSupport", 0) or 0)
            hold_head = float(hold.get("ringHeadSupport", 0) or 0)
            if hold_ring < .80 or hold_head < .75:
                continue
            if gap < .065 or gap > .115:
                continue
            dx = float(tap.get("x", 0)) - float(hold.get("x", 0))
            dy = (float(tap.get("judgeY", .822222)) - float(hold.get("judgeY", .822222))) * 720
            if math.hypot(dx, dy) > 65:
                continue
            end = float(hold["time"]) + duration
            tap["time"] = round(max(0., float(tap["time"]) - frame), 5)
            tap["sourceTimingAdjustment"] = round(-frame, 5)
            if tap.get("counterPinnedTime") is not None:
                tap["counterPinnedTime"] = tap["time"]
            hold["time"] = round(float(hold["time"]) + frame, 5)
            hold["duration"] = round(end - float(hold["time"]), 5)
            hold["sourceTimingAdjustment"] = round(frame, 5)
            used.update((id(tap), id(hold)))
            changes.append({
                "tapTime": tap["time"], "holdTime": hold["time"],
                "tapEvidence": tap.get("evidence"), "holdEvidence": hold.get("evidence"),
                "adjustmentSeconds": round(frame, 5),
                "holdEndPreserved": round(end, 5),
                "screenDistance": round(math.hypot(dx, dy), 2),
            })
            break
    return {"adjustedPairs": len(changes), "changes": changes}
