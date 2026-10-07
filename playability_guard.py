"""Apply minimal video-frame nudges when autoplay exposes touch stealing."""
from __future__ import annotations


def nudge_early_contacts(events, bad_notes, fps=30.0, max_shift_frames=1):
    """Move a bad later head by one source frame when an earlier touch stole it.

    The renderer-based session review reports the chart note id and the time
    of the early touch that captured it. Very strongly measured note heads are
    left untouched; uncertain heads may move by at most one source frame per
    pass, preserving a hold's original tail time.
    """
    emitted = [e for e in sorted(events, key=lambda q: float(q.get("time", 0)))
               if not (e.get("isFake") and not e.get("path")
                       and e.get("evidence") == "video-counter-and-fresh-source-hit-ring")]
    frame = 1.0 / max(1.0, float(fps))
    changed = []
    for bad in sorted(bad_notes, key=lambda q: float(q.get("t", 0))):
        index = int(bad.get("id", -1))
        if not 0 <= index < len(emitted):
            continue
        event = emitted[index]
        if event.get("isFake") or int(event.get("type", 0)) == 2:
            continue
        contact = event.get("sourceContactTime")
        ring = float(event.get("ringContactSupport", 0) or 0)
        head = float(event.get("ringHeadSupport", 0) or 0)
        # A clearly measured ring and head outrank the synthetic touch test.
        if contact is not None and ring >= .90 and head >= .90:
            continue
        old = float(event.get("time", bad.get("t", 0)))
        if float(bad.get("judged", old)) >= old - 1e-4:
            continue
        end = old + float(event.get("duration", 0) or 0)
        shift = frame * max(1, int(max_shift_frames))
        event["previousHeadTime"] = old
        event["time"] = round(old + shift, 5)
        event["playabilityTimingAdjustment"] = round(shift, 5)
        if event.get("duration", 0) > 0:
            event["duration"] = round(max(.015, end - event["time"]), 5)
        changed.append({"id": index, "from": round(old, 5),
                        "to": event["time"], "judgedEarlyAt": bad.get("judged"),
                        "ringContactSupport": ring, "ringHeadSupport": head})
    return changed
