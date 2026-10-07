"""Resolve duplicate endpoint detections on fast moving holds.

Some video trackers follow both ends of one capsule as separate candidates.
When those passes share a source contact, release, track identity and both
measured endpoint paths, keep the pass with the stronger head contact.
"""
from __future__ import annotations

import math

import numpy as np


def _path(event, name):
    value = np.asarray(event.get(name) or [], dtype=float)
    if value.ndim != 2 or len(value) < 4 or value.shape[1] < 3:
        return None
    value = value[np.argsort(value[:, 0], kind="stable")]
    _, indices = np.unique(value[:, 0], return_index=True)
    value = value[np.sort(indices)]
    return value if len(value) >= 4 else None


def _agreement(a, b, end):
    lo = max(float(a[0, 0]), float(b[0, 0]), end - .20)
    hi = min(float(a[-1, 0]), float(b[-1, 0]), end + .025)
    if hi - lo < .06:
        return None
    times = np.linspace(lo, hi, max(5, int(math.ceil((hi - lo) * 30)) + 1))
    aa = np.column_stack([np.interp(times, a[:, 0], a[:, k]) for k in (1, 2)])
    bb = np.column_stack([np.interp(times, b[:, 0], b[:, k]) for k in (1, 2)])
    errors = np.linalg.norm(aa - bb, axis=1)
    return float(np.median(errors)), float(np.quantile(errors, .9))


def collapse_duplicate_hold_endpoint_passes(events, *, flow_deviation=50.0):
    """Remove a weaker duplicate pass when both ends trace one real capsule."""
    removed = set()
    findings = []
    for i, a in enumerate(events):
        if i in removed or a.get("isFake") or a.get("visualOnly"):
            continue
        da = float(a.get("duration", 0) or 0)
        ca = a.get("sourceContactTime")
        if da <= .02 or ca is None or a.get("track") is None:
            continue
        if abs(float(a.get("speed", 960) or 960) - 960) <= flow_deviation:
            continue
        ah, at = _path(a, "path"), _path(a, "tailPath")
        if ah is None or at is None:
            continue
        ea = float(a.get("time", 0)) + da
        for j in range(i + 1, len(events)):
            b = events[j]
            if j in removed or b.get("isFake") or b.get("visualOnly"):
                continue
            if b.get("track") != a.get("track") or int(b.get("type", 0)) != int(a.get("type", 0)):
                continue
            db = float(b.get("duration", 0) or 0)
            cb = b.get("sourceContactTime")
            if db <= .02 or cb is None or abs(float(cb) - float(ca)) > .025:
                continue
            tb = float(b.get("time", 0))
            eb = tb + db
            if not .03 <= tb - float(a.get("time", 0)) <= .20 or abs(eb - ea) > .045:
                continue
            head_a = float(a.get("ringHeadSupport", 0) or 0)
            head_b = float(b.get("ringHeadSupport", 0) or 0)
            contact_a = float(a.get("ringContactSupport", 0) or 0)
            contact_b = float(b.get("ringContactSupport", 0) or 0)
            if head_a < .65 or head_b > .55 or head_a - head_b < .20:
                continue
            if contact_a < .55 or contact_b < .35:
                continue
            bh, bt = _path(b, "path"), _path(b, "tailPath")
            if bh is None or bt is None:
                continue
            head_fit = _agreement(ah, bh, min(ea, eb))
            tail_fit = _agreement(at, bt, min(ea, eb))
            if (head_fit is None or tail_fit is None
                    or head_fit[0] > 14 or head_fit[1] > 24
                    or tail_fit[0] > 14 or tail_fit[1] > 24):
                continue
            # Both endpoints are already represented by the stronger pass.
            removed.add(j)
            findings.append({
                "track": a.get("track"),
                "keptTime": float(a.get("time", 0)),
                "removedTime": tb,
                "headPathMedianPx": round(head_fit[0], 3),
                "tailPathMedianPx": round(tail_fit[0], 3),
                "sharedReleaseDelta": round(eb - ea, 5),
                "headContactSupport": [head_a, head_b],
            })
            break
    if removed:
        events[:] = [event for index, event in enumerate(events) if index not in removed]
    return {"removedDuplicateEndpointPasses": len(removed), "findings": findings}
