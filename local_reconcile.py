"""Refine local judgement allocation from counter frames and measured contacts."""
from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter


_WEAK_EVIDENCE = {
    "full-screen-decorative-track",
    "independent-faint-source-approach",
    "faint-hollow-moving-decoration",
    "video-counter-boundary-repair",
}
_NEVER_PROMOTE = {"source-hollow-single-frame-sprite", "non-note-particle-fragment"}


def _atoms(e):
    if e.get("isFake") or e.get("visualOnly"):
        return []
    start = float(e.get("time", 0.0))
    out = [start]
    duration = float(e.get("duration", 0.0))
    if duration > 0:
        out.append(start + duration)
    return out


def _interval_index(t, bounds):
    i = bisect.bisect_left(bounds, float(t) - 1e-6) - 1
    return i if 0 <= i < len(bounds) - 1 and bounds[i] - 1e-6 < t <= bounds[i + 1] + 1e-6 else None


def _interval_atoms(e, bounds):
    return Counter(i for t in _atoms(e) if (i := _interval_index(t, bounds)) is not None)


def _score(e, promote=False):
    frames = float(e.get("frames", 0) or 0)
    ring = float(e.get("ringContactSupport", 0) or 0)
    head = float(e.get("ringHeadSupport", 0) or 0)
    hit = float(e.get("headHitSupport", 0) or 0)
    track = 1.3 if e.get("targetTrack") is not None else 0.0
    path = min(2.0, math.log1p(frames) * .34)
    continuity = .8 if len(e.get("path", [])) >= 5 else 0.0
    weak = 1.8 if e.get("evidence", "") in _WEAK_EVIDENCE else 0.0
    return ring * 5.0 + head * 3.0 + hit * 4.0 + track + path + continuity - weak


def _l1_delta(residual, before, after):
    """Change in total local counter error after replacing event atoms."""
    delta = after.copy()
    delta.subtract(before)
    return sum(abs(residual[i] - change) - abs(residual[i]) for i, change in delta.items())


def _times_in_residual(e, residual, bounds, sign, include_fake=False):
    times = ([float(e.get("time", 0.0))]
             + ([float(e.get("time", 0.0)) + float(e.get("duration", 0.0))]
                if float(e.get("duration", 0.0)) > 0 else [])) if include_fake else _atoms(e)
    return [t for t in times
            if (i := _interval_index(t, bounds)) is not None
            and (residual[i] > 0 if sign > 0 else residual[i] < 0)]


def _shifted_atoms(e, bounds, target, kind):
    old = _interval_atoms(e, bounds)
    t = float(e.get("time", 0.0))
    duration = float(e.get("duration", 0.0))
    if kind == "trim-tail":
        if duration <= 0:
            return None
        new = Counter(i for i in [_interval_index(t, bounds)] if i is not None)
        return old, new
    if kind == "head":
        # Preserve explicit source-frame repairs and strong hit-ring contacts.
        # The OCR counter can update on the adjacent video frame; that is not
        # evidence that the measured note itself happened at the boundary.
        if e.get("counterRequired") and e.get("repairCounterInterval"):
            start, end = map(float, e["repairCounterInterval"])
            if not start - 1e-6 < target <= end + 1e-6:
                return None
        if e.get("counterPinnedTime") is not None and abs(target - float(e["counterPinnedTime"])) > .002:
            return None
        if (e.get("sourceContactTime") is not None
                and float(e.get("ringContactSupport", 0) or 0) >= .75
                and float(e.get("ringHeadSupport", 0) or 0) >= .75
                and abs(target - float(e["sourceContactTime"])) > .025):
            return None
        if duration and target + duration < t + .015:
            return None
        new_times = [target]
        if duration > 0:
            new_times.append(target + duration)
    else:
        new_duration = target - t
        if not .015 <= new_duration <= 3.5:
            return None
        new_times = [t, target]
    new = Counter(i for tm in new_times if (i := _interval_index(tm, bounds)) is not None)
    return old, new


def _observed_tail_contact(e):
    """Estimate when a measured hold tail reaches its judgement point.

    Counter fitting can move a hold head later while retaining an earlier
    release, collapsing a physically observed hold into a few frames.  Use
    this conservative floor only for a tracked, near-vertical capsule whose
    tail is still well above its judgement point at the proposed release.
    Rotating lines and tracked moving targets need their own geometry and are
    deliberately left to the existing fitters.
    """
    duration = float(e.get("duration", 0.0) or 0.0)
    if duration <= 0 or e.get("isFake") or e.get("visualOnly"):
        return None
    if e.get("targetTrack") is not None:
        return None
    if e.get("holdEndEvidence") != "stationary-capsule-tail-and-disappearance":
        return None
    if abs(float(e.get("rotation", 90.0)) - 90.0) > 8.0:
        return None
    rotation_path = e.get("holdRotationPath") or []
    angles = [float(q[1]) for q in rotation_path if len(q) >= 2]
    if len(angles) >= 3 and max(angles) - min(angles) > 10.0:
        return None
    path = sorted((q for q in (e.get("tailPath") or []) if len(q) >= 3),
                  key=lambda q: float(q[0]))
    if len(path) < 8:
        return None

    start = float(e.get("time", 0.0))
    end = start + duration
    if end < float(path[0][0]) or end > float(path[-1][0]) + 1e-6:
        return None
    target_y = float(e.get("judgeY", 0.822222)) * 720.0

    def y_at(t):
        for a, b in zip(path, path[1:]):
            ta, tb = float(a[0]), float(b[0])
            if ta <= t <= tb and tb > ta:
                ratio = (t - ta) / (tb - ta)
                return float(a[2]) + ratio * (float(b[2]) - float(a[2]))
        return None

    tail_y = y_at(end)
    if tail_y is None:
        return None
    # Require clear physical separation; small residuals are within capsule
    # and video-frame measurement error and should not retime a hold.
    if target_y - tail_y < 65.0:
        return None

    # Prefer a measured crossing after the fitted release. If the video ends
    # while the tail is still approaching, extrapolate from the last few
    # observed source-frame segments.
    for a, b in zip(path, path[1:]):
        ta, tb = float(a[0]), float(b[0])
        ya, yb = float(a[2]), float(b[2])
        if tb <= end or tb <= ta or ya > target_y or yb < target_y:
            continue
        contact = ta + (target_y - ya) * (tb - ta) / max(1e-6, yb - ya)
        return contact if end + .08 <= contact <= end + .40 else None

    observed = [q for q in path if float(q[0]) >= end and float(q[0]) <= end + .30]
    if len(observed) < 3:
        return None
    slopes = [(float(b[2]) - float(a[2])) / (float(b[0]) - float(a[0]))
              for a, b in zip(observed, observed[1:])
              if float(b[0]) > float(a[0])]
    positive = [v for v in slopes if 250.0 <= v <= 2400.0]
    if len(positive) < 2:
        return None
    positive.sort()
    velocity = positive[len(positive) // 2]
    last = observed[-1]
    remaining = target_y - float(last[2])
    if remaining <= 0:
        return None
    contact = float(last[0]) + remaining / velocity
    if not end + .08 <= contact <= end + .40:
        return None
    return contact


def _snap(e, target, kind):
    if kind == "trim-tail":
        old_duration = float(e.get("duration", 0.0))
        e["previousDuration"] = old_duration
        e["duration"] = 0.0
        e["tailTimingAdjustment"] = round(-old_duration, 5)
        e["counterTailRemoved"] = True
        return
    if kind == "head":
        old = float(e["time"])
        e["previousHeadTime"] = old
        e["time"] = round(max(0.0, target), 5)
        e["timingAdjustment"] = round(e["time"] - old, 5)
        if e.get("sourceTime") is not None:
            e["sourceTime"] = e["time"]
    else:
        old_duration = float(e["duration"])
        e["duration"] = round(target - float(e["time"]), 5)
        e["tailTimingAdjustment"] = round(e["duration"] - float(e.get("fitBaseDuration", old_duration)), 5)


def _joint_variant_options(e, *, duplicate=False):
    """Return conservative alternate states for a short local counter fit.

    This is intentionally narrower than the normal note classifier: it only
    revisits events with measurable source geometry and keeps the change cost
    proportional to the independent visual evidence.
    """
    options = []
    if e.get("isFake"):
        if e.get("visualOnly") or e.get("evidence") in _NEVER_PROMOTE:
            return options
        ring = float(e.get("ringContactSupport", 0) or 0)
        head = float(e.get("ringHeadSupport", 0) or 0)
        hit = float(e.get("headHitSupport", 0) or 0)
        path = e.get("path") or []
        has_contact = (ring + head >= .55 or e.get("counterRequired")
                       or (e.get("sourceContactTime") is not None and ring >= .5))
        has_geometry = (len(path) >= 5 and e.get("targetTrack") is not None)
        has_capsule = (len(path) >= 8 and e.get("bodyEvidence")
                       and e.get("holdEndEvidence"))
        if has_contact and (has_geometry or has_capsule):
            options.append(("promote", None,
                            max(.45, 4.5 - .35 * _score(e))))
        return options

    duration = float(e.get("duration", 0) or 0)
    if (duration > .015 and duration <= .40
            and not e.get("holdEndEvidence")
            and float(e.get("ringHeadSupport", 0) or 0) < .5
            and len(e.get("path") or []) < 12
            and not e.get("visualOnly")):
        # Keep the independently-supported head and drop only a tail for which
        # the detector did not find a capsule end/disappearance.
        options.append(("trim-tail", 0.0,
                        1.15 + .10 * min(10.0, _score(e))))
    if duplicate:
        # A same-track, same-time, same-length capsule pair is usually two
        # detector passes over one physical note. Only expose the lower-ranked
        # pass as removable; the combo fit still has to justify choosing it.
        options.append(("dedupe", None,
                        .65 + .12 * min(10.0, _score(e))))
    return options


def _joint_counter_refine(events, bounds, expected, max_rounds=12):
    """Resolve adjacent counter residuals with a bounded local assignment.

    Single-event boundary moves cannot solve cases where a weak capsule tail,
    a missed but visible note, and duplicated detector passes overlap across
    consecutive OCR intervals. For each connected residual cluster, this
    searches only evidence-backed promote/trim/dedupe choices and accepts a
    combination only when it reduces total local counter error.
    """
    actions = []
    for _ in range(max_rounds):
        counts = [0] * len(expected)
        for e in events:
            for i, n in _interval_atoms(e, bounds).items():
                counts[i] += n
        residual = [expected[i] - counts[i] for i in range(len(expected))]
        bad = [i for i, v in enumerate(residual) if v]
        if not bad:
            break

        # Adjacent OCR windows form one allocation problem; nearby residual
        # windows are joined only when their gap is short enough that a hold
        # endpoint could bridge them.
        clusters = []
        for i in bad:
            if clusters and bounds[i] - bounds[clusters[-1][-1] + 1] <= .20:
                clusters[-1].append(i)
            else:
                clusters.append([i])

        changed_this_round = False
        for cluster in clusters:
            start, end = bounds[cluster[0]], bounds[cluster[-1] + 1]
            lo, hi = start - .20, end + .60
            duplicate_indices = set()
            duplicate_groups = []
            holds = [(idx, e) for idx, e in enumerate(events)
                     if not e.get("isFake") and float(e.get("duration", 0) or 0) > 0
                     and e.get("track") is not None
                     and lo <= float(e.get("time", 0)) <= hi]
            for pos, (ia, a) in enumerate(holds):
                for ib, b in holds[pos + 1:]:
                    if (a.get("track") == b.get("track")
                            and abs(float(a["time"]) - float(b["time"])) <= .004
                            and abs(float(a.get("duration", 0)) - float(b.get("duration", 0))) <= .02
                            and abs(float(a.get("x", 0)) - float(b.get("x", 0))) <= 12):
                        # Preserve the stronger pass, resolving ties by stable
                        # event order so output remains deterministic.
                        loser = min((ia, ib), key=lambda idx: (_score(events[idx]), -idx))
                        duplicate_indices.add(loser)

            candidates = []
            for idx, e in enumerate(events):
                t = float(e.get("time", 0) or 0)
                d = float(e.get("duration", 0) or 0)
                if not lo <= t <= hi and not lo <= t + d <= hi:
                    continue
                opts = _joint_variant_options(e, duplicate=idx in duplicate_indices)
                if opts:
                    candidates.append((idx, e, opts))
            if not candidates or len(candidates) > 14:
                continue

            baseline = {idx: _interval_atoms(e, bounds)
                        for idx, e, _ in candidates}
            affected = set(cluster)
            for atoms in baseline.values():
                affected.update(atoms)
            for _, e, opts in candidates:
                for kind, value, _ in opts:
                    if kind == "promote":
                        t = float(e.get("time", 0) or 0)
                        d = float(e.get("duration", 0) or 0)
                        for q in (t, t + d if d > 0 else None):
                            if q is not None and (i := _interval_index(q, bounds)) is not None:
                                affected.add(i)
                    elif kind == "trim-tail" and (i := _interval_index(
                            float(e.get("time", 0)) + float(e.get("duration", 0)), bounds)) is not None:
                        affected.add(i)
            affected = sorted(affected)
            before_l1 = sum(abs(residual[i]) for i in affected)
            best = None
            option_sets = [[(None, None, 0.0)] + opts for _, _, opts in candidates]
            for choices in itertools.product(*option_sets):
                if not any(kind for kind, _, _ in choices):
                    continue
                delta = Counter()
                penalty = 0.0
                for (idx, e, _), (kind, value, cost) in zip(candidates, choices):
                    if kind is None:
                        continue
                    penalty += cost
                    old = baseline[idx]
                    delta.subtract(old)
                    if kind == "promote":
                        t = float(e.get("time", 0) or 0)
                        d = float(e.get("duration", 0) or 0)
                        new = Counter(i for q in ([t, t + d] if d > 0 else [t])
                                      if (i := _interval_index(q, bounds)) is not None)
                        delta.update(new)
                    elif kind == "trim-tail":
                        if (i := _interval_index(float(e.get("time", 0)), bounds)) is not None:
                            delta[i] += 1
                    # Dedupe contributes no replacement atoms.
                after_l1 = sum(abs(residual[i] - delta[i]) for i in affected)
                if after_l1 >= before_l1:
                    continue
                objective = 20.0 * after_l1 + penalty
                if best is None or objective < best[0]:
                    best = (objective, after_l1, choices)

            if best is None:
                continue
            _, _, choices = best
            for (idx, e, _), (kind, value, _) in zip(candidates, choices):
                if kind is None:
                    continue
                if kind == "promote":
                    e["isFake"] = False
                    e["judgementEvidence"] = "measured-joint-local-counter-reconciliation"
                elif kind == "trim-tail":
                    old = float(e.get("duration", 0) or 0)
                    e["previousDuration"] = old
                    e["duration"] = 0.0
                    e["tailTimingAdjustment"] = round(-old, 5)
                    e["counterTailRemoved"] = True
                elif kind == "dedupe":
                    e["isFake"] = True
                    e["judgementEvidence"] = "same-track-duplicate-pass"
                e["counterReconciled"] = True
                actions.append({"action": kind, "track": e.get("track"),
                                "time": float(e.get("time", 0) or 0),
                                "improvement": round(before_l1 - best[1], 3)})
                changed_this_round = True
        if not changed_this_round:
            break
    return actions


def reconcile(events, allocation, max_passes=96):
    """Minimize local combo residuals with evidence-ranked joint adjustments.

    A change is accepted only when it reduces the sum of per-interval errors.
    This prevents a boundary correction from moving an error into a previously
    correct interval, which the former greedy pass could do.
    """
    cumulative = allocation.get("cumulative") or []
    if not cumulative:
        return {"changed": 0, "resolved": 0, "remaining": 0, "passes": 0}
    bounds = [0.0] + [float(q["time"]) for q in cumulative]
    expected = [int(cumulative[i]["videoCombo"] - (cumulative[i - 1]["videoCombo"] if i else 0))
                for i in range(len(cumulative))]
    changed = 0
    audit = []

    for pass_no in range(max_passes):
        actual = [0] * len(expected)
        for e in events:
            for i, count in _interval_atoms(e, bounds).items():
                actual[i] += count
        residual = [expected[i] - actual[i] for i in range(len(expected))]
        if not any(residual):
            break

        actions = []
        # Try moving a measured head or tail across an adjacent readable-frame
        # boundary. Re-evaluate both ends for a hold before accepting a move.
        for e in events:
            if e.get("isFake") or e.get("visualOnly"):
                continue
            duration = float(e.get("duration", 0.0))
            physical_tail = _observed_tail_contact(e)
            contacts = [(float(e["time"]), "head")]
            if duration > 0:
                contacts.append((float(e["time"]) + duration, "tail"))
            for contact, kind in contacts:
                near = bisect.bisect_left(bounds, contact)
                for boundary_i in (near - 1, near):
                    if not 0 < boundary_i < len(bounds) - 1:
                        continue
                    boundary = bounds[boundary_i]
                    if abs(contact - boundary) > .084:
                        continue
                    target = boundary - .001 if contact > boundary else boundary + .001
                    if (kind == "tail" and physical_tail is not None
                            and target < physical_tail - .015):
                        continue
                    pair = _shifted_atoms(e, bounds, target, kind)
                    if pair is None:
                        continue
                    before, after = pair
                    delta = _l1_delta(residual, before, after)
                    if delta < 0:
                        actions.append((delta, 0, abs(target - contact) * 4 - _score(e),
                                        "move", e, target, kind))

            # A physical tail contact can be more than two counter frames
            # after the fitted release. Consider it directly when local combo
            # evidence also improves; this repairs holds that were shortened
            # by a delayed head fit without guessing from a fixed timestamp.
            contact = float(e.get("time", 0.0)) + duration
            if (physical_tail is not None and physical_tail > contact + .084):
                pair = _shifted_atoms(e, bounds, physical_tail, "tail")
                if pair is not None:
                    before, after = pair
                    delta = _l1_delta(residual, before, after)
                    if delta < 0:
                        actions.append((delta, 0,
                                        abs(physical_tail - contact) * 4 - _score(e),
                                        "move", e, physical_tail, "tail"))

        promotions = []
        demotions = []
        tail_trims = []
        # Promote a rejected observation only when its own measured path or
        # contact support makes the judgement credible and local counts need it.
        # Promotion and demotion are paired below so the local pass preserves
        # the independently measured end-of-chart combo.
        for e in events:
            if not e.get("isFake") or e.get("visualOnly") or e.get("evidence") in _NEVER_PROMOTE:
                continue
            support = float(e.get("ringContactSupport", 0) or 0) + float(e.get("ringHeadSupport", 0) or 0)
            if support < .35 and len(e.get("path", [])) < 4 and e.get("targetTrack") is None:
                continue
            atoms = Counter(i for tm in [float(e.get("time", 0.0))] +
                            ([float(e["time"]) + float(e.get("duration", 0.0))]
                             if float(e.get("duration", 0.0)) > 0 else [])
                            if (i := _interval_index(tm, bounds)) is not None)
            if any(residual[i] > 0 for i in atoms):
                promotions.append((e, atoms))

        # If an interval has too many judgements, consider the weakest supported
        # event as the removal half of a total-preserving exchange.
        for e in events:
            if e.get("isFake") or e.get("visualOnly"):
                continue
            atoms = _interval_atoms(e, bounds)
            if any(residual[i] < 0 for i in atoms):
                demotions.append((e, atoms))

            # A very short detected capsule can be a tap with a trailing glow
            # mistaken for a hold body. If the local video counter proves one
            # excess judgement at its endpoint, allow removing only the tail;
            # retain the independently supported head and note artwork.
            duration = float(e.get("duration", 0.0))
            weak_head = (float(e.get("ringHeadSupport", 0) or 0) < .50
                         and float(e.get("headHitSupport", 0) or 0) < .30)
            weak_track = e.get("evidence") in _WEAK_EVIDENCE or e.get("evidence") == "full-screen-decorative-track"
            if .015 < duration <= .12 and weak_head and weak_track:
                tail_i = _interval_index(float(e.get("time", 0.0)) + duration, bounds)
                if tail_i is not None and residual[tail_i] < 0:
                    pair = _shifted_atoms(e, bounds, 0.0, "trim-tail")
                    if pair is not None:
                        removed = pair[0].copy()
                        removed.subtract(pair[1])
                        removed = Counter({i: n for i, n in removed.items() if n > 0})
                        tail_trims.append((e, removed))

        for promoted, add_atoms in promotions:
            add_count = sum(add_atoms.values())
            add_times = _times_in_residual(promoted, residual, bounds, 1, include_fake=True)
            for demoted, remove_atoms in demotions:
                if promoted is demoted or sum(remove_atoms.values()) != add_count:
                    continue
                remove_times = _times_in_residual(demoted, residual, bounds, -1)
                if not add_times or not remove_times or min(abs(a - b) for a in add_times for b in remove_times) > .5:
                    continue
                delta = _l1_delta(residual, remove_atoms, add_atoms)
                if delta < 0:
                    actions.append((delta, 1, _score(demoted) - _score(promoted),
                                    "exchange", promoted, demoted, None))
            for hold, remove_atoms in tail_trims:
                tail_time = float(hold.get("time", 0.0)) + float(hold.get("duration", 0.0))
                if not add_times or min(abs(tail_time - a) for a in add_times) > .5:
                    continue
                delta = _l1_delta(residual, remove_atoms, add_atoms)
                if delta < 0 and sum(remove_atoms.values()) == add_count:
                    actions.append((delta, 1, _score(hold) - _score(promoted),
                                    "trim-exchange", promoted, hold, None))

        if not actions:
            break
        delta, _, _, action, e, target, kind = min(actions, key=lambda q: (q[0], q[1], q[2]))
        if action == "move":
            _snap(e, target, kind)
        elif action == "exchange":
            e["isFake"] = False
            e["judgementEvidence"] = "measured-local-counter-reconciliation"
            e["counterReconciled"] = True
            target["isFake"] = True
            target["counterReconciled"] = True
        elif action == "trim-exchange":
            e["isFake"] = False
            e["judgementEvidence"] = "measured-local-counter-reconciliation"
            e["counterReconciled"] = True
            _snap(target, 0.0, "trim-tail")
            target["counterReconciled"] = True
        audit.append({"action": action, "track": e.get("track"), "time": float(e.get("time", 0.0)),
                      "pairedTrack": target.get("track") if action in ("exchange", "trim-exchange") else None,
                      "improvement": -delta})
        changed += 1

    joint_actions = _joint_counter_refine(events, bounds, expected)
    audit.extend(joint_actions)
    changed += len(joint_actions)

    final = [0] * len(expected)
    for e in events:
        for i, count in _interval_atoms(e, bounds).items():
            final[i] += count
    remaining = [{"from": bounds[i], "to": bounds[i + 1], "expected": expected[i],
                  "actual": final[i], "difference": expected[i] - final[i]}
                 for i in range(len(expected)) if final[i] != expected[i]]
    return {"changed": changed, "resolved": len(expected) - len(remaining),
            "remaining": len(remaining), "passes": len(audit), "unresolved": remaining,
            "actions": audit}


def align_weak_heads(events, observations, fps=30.0, max_shift=0.10,
                     min_improvement=0.75):
    """Align weakly localized heads to a persistent source-counter transition.

    Physical ring/head contacts remain pinned by the main fitter. This pass is
    for uncertain approaches whose selected time precedes the observed combo
    step by one or two video frames. Candidate shifts are limited to the local
    source-frame window and must lower the absolute counter error.
    """
    frame = 1.0 / max(1.0, float(fps))
    samples = []
    for q in observations or []:
        try:
            t, combo = float(q["time"]), int(q["combo"])
        except (KeyError, TypeError, ValueError):
            continue
        confidence = float(q.get("confidence", max(0.0, 100.0 * (1.0 - float(q.get("error", .1))))))
        if confidence < 90 or combo < 0:
            continue
        samples.append((t, combo, min(1.0, confidence / 100.0)))
    samples.sort()
    if not samples:
        return {"changed": 0, "actions": []}

    atoms = sorted((t, i) for i, e in enumerate(events)
                   for t in _atoms(e))
    atom_times = [q[0] for q in atoms]
    predictions = [bisect.bisect_right(atom_times, t + 1e-7) for t, _, _ in samples]
    errors = [predictions[i] - q[1] for i, q in enumerate(samples)]
    max_frames = max(1, int(round(float(max_shift) / frame)))
    actions = []

    for index, e in enumerate(events):
        if e.get("isFake") or e.get("visualOnly") or e.get("counterRequired"):
            continue
        if e.get("evidence") in {
            "faint-hollow-moving-decoration",
            "source-hollow-single-frame-sprite",
            "non-note-particle-fragment",
        }:
            continue
        # The cumulative counter can refine tap timing, but moving a hold head
        # also moves its release in the native chart.  Hold heads and tails
        # are measured independently elsewhere, so leave both endpoints alone.
        if float(e.get("duration", 0.0) or 0.0) > 0:
            continue
        if e.get("counterPinnedTime") is not None or e.get("counterPinnedContact"):
            continue
        # Preserve timing adjustments inserted by the independent contact
        # spacing guard; undoing one can make a touch be captured by the
        # neighbouring hold head again.
        if e.get("sourceTimingAdjustment") is not None:
            continue
        ring = float(e.get("ringContactSupport", 0) or 0)
        head = float(e.get("ringHeadSupport", 0) or 0)
        hit = float(e.get("headHitSupport", 0) or 0)
        has_independent_support = (
            e.get("targetTrack") is not None
            or ring >= .25
            or head >= .25
            or hit >= .25
            or bool(e.get("approachEvidence"))
            or len(e.get("path", [])) >= 4
        )
        if not has_independent_support:
            continue
        weak = ((head < .50 and hit < .35)
                or (e.get("evidence") == "full-screen-decorative-track" and head < .70))
        if not weak or (ring >= .75 and head >= .75):
            continue
        base = float(e.get("time", 0.0))
        duration = float(e.get("duration", 0.0))
        old_times = [base] + ([base + duration] if duration > 0 else [])
        candidate_indices = [i for i, (t, _, _) in enumerate(samples)
                             if base - max_shift - .01 <= t <= base + max_shift + duration + .01]
        if not candidate_indices:
            continue
        base_cost = sum(samples[i][2] * abs(errors[i]) for i in candidate_indices)
        options = []
        for steps in range(-max_frames, max_frames + 1):
            if steps == 0:
                continue
            new = max(0.0, base + steps * frame)
            if duration > 0 and new + duration < base + .015:
                continue
            new_times = [new] + ([new + duration] if duration > 0 else [])
            cost = 0.0
            changed_samples = 0
            for j in candidate_indices:
                t, _, weight = samples[j]
                before = sum(x <= t + 1e-7 for x in old_times)
                after = sum(x <= t + 1e-7 for x in new_times)
                err = errors[j] + after - before
                cost += weight * abs(err)
                changed_samples += int(after != before)
            improvement = base_cost - cost
            if improvement >= min_improvement and changed_samples:
                options.append((cost, abs(steps), steps, improvement, new))
        if not options:
            continue
        _, _, _, improvement, new = min(options)
        old = float(e["time"])
        e["previousHeadTime"] = old
        e["time"] = round(new, 5)
        e["sourceTime"] = e["time"]
        e["counterTimingReconciled"] = True
        e["counterTimingAdjustment"] = round(e["time"] - old, 5)
        actions.append({"time": old, "newTime": e["time"],
                        "adjustment": e["counterTimingAdjustment"],
                        "evidence": e.get("evidence"),
                        "counterErrorImprovement": round(improvement, 3)})
        # Update the counter trace so later independent candidates are scored
        # against the already accepted timing correction.
        old_items = [(t, i) for t, i in atoms if i != index]
        new_items = old_items + [(t, index) for t in new_times]
        new_items.sort()
        atoms = new_items
        atom_times = [q[0] for q in atoms]
        predictions = [bisect.bisect_right(atom_times, t + 1e-7) for t, _, _ in samples]
        errors = [predictions[i] - q[1] for i, q in enumerate(samples)]

    return {"changed": len(actions), "actions": actions}
