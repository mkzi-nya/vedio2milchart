"""Keep local video-counter fitting from regressing an already coherent chart.

The per-frame fitter is allowed to propose event and timing changes, but each
proposal is checked against the source counter curve before it replaces the
current observation.  When the complete fit improves the curve, it is kept.
When it regresses a chart that already has the right final judgement count, we
retain only counter-supported, count-preserving changes that improve the
independent frame samples.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


_FIT_FIELDS = (
    "time", "duration", "isFake", "sourceTime", "timingAdjustment",
    "tailTimingAdjustment", "previousHeadTime", "previousDuration",
    "judgementEvidence", "counterReconciled", "counterTailRemoved",
)


def pin_drifted_contacts(events: list[dict[str, Any]], min_drift: float = .04,
                        max_drift: float = .12, min_support: float = .75) -> list[int]:
    """Pin strong measured contacts that a counter refit moved too far.

    A refit can move an event that matched its source contact in the previous
    pass, especially after another event is pinned and changes the local
    allocation.  Rechecking after every pass keeps the measured ring contact
    from drifting out of alignment while still leaving weak observations to
    the counter fitter.
    """
    pinned = []
    for index, event in enumerate(events):
        contact = event.get("sourceContactTime")
        if (event.get("isFake") or contact is None
                or event.get("counterPinnedContact")):
            continue
        ring = float(event.get("ringContactSupport", 0) or 0)
        head = float(event.get("ringHeadSupport", 0) or 0)
        drift = abs(float(event.get("time", 0)) - float(contact))
        if ring >= min_support and head >= min_support and min_drift < drift <= max_drift:
            event["counterPinnedContact"] = True
            pinned.append(index)
    return pinned


def _atoms(event: dict[str, Any]) -> list[float]:
    if event.get("isFake") or event.get("visualOnly"):
        return []
    start = float(event.get("time", 0.0))
    duration = float(event.get("duration", 0.0))
    return [start, start + duration] if duration > 0 else [start]


def _samples(observations: list[dict[str, Any]]):
    # These are the same broad reliability limits used by the fitter. The
    # margin and glyph error still weight each observation independently.
    selected = [q for q in observations
                if (q.get("error", 1) < .085 and q.get("margin", 0) > .025)
                or (q.get("error", 1) < .045 and q.get("margin", 0) > .020)]
    if not selected:
        return None
    times = np.asarray([float(q["time"]) for q in selected])
    counts = np.asarray([int(q["combo"]) for q in selected])
    weights = np.asarray([
        max(.15, min(1., (float(q["margin"]) - .015) / .075))
        * max(.2, min(1., (.15 - float(q["error"])) / .12))
        for q in selected
    ])
    return times, counts, weights


def _event_vector(event: dict[str, Any], times: np.ndarray) -> np.ndarray:
    result = np.zeros(len(times), dtype=np.int16)
    for atom in _atoms(event):
        result += times >= atom
    return result


def _curve(events: list[dict[str, Any]], times: np.ndarray) -> np.ndarray:
    atoms = sorted(atom for event in events for atom in _atoms(event))
    return np.searchsorted(atoms, times, side="right")


def _loss(curve: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * np.abs(counts - curve)))


def _contact_safe(seed: list[dict[str, Any]], fitted: list[dict[str, Any]]) -> bool:
    """A counter fit cannot overrule a clearly measured note-head contact."""
    for old, new in zip(seed, fitted):
        if old.get("isFake") or new.get("isFake"):
            continue
        contact = old.get("sourceContactTime")
        strong = (float(old.get("ringContactSupport", 0) or 0) >= .75
                  and float(old.get("ringHeadSupport", 0) or 0) >= .75)
        if (strong and contact is not None
                and abs(float(new.get("time", 0)) - float(contact)) > .025
                and abs(float(new.get("time", 0)) - float(old.get("time", 0))) > .025):
            return False
    return True


def _supported_promotion(event: dict[str, Any]) -> bool:
    """Require visible note or hit evidence before a counter-only promotion."""
    if event.get("evidence") in {
        "source-hollow-single-frame-sprite", "non-note-particle-fragment",
    }:
        return False
    ring = float(event.get("ringContactSupport", 0) or 0)
    head = float(event.get("ringHeadSupport", 0) or 0)
    hit = float(event.get("headHitSupport", 0) or 0)
    return bool(event.get("counterRequired") or event.get("approachEvidence")
                or ring >= .35 or head >= .3 or hit >= .25
                or event.get("targetTrack") is not None
                or len(event.get("path", [])) >= 5)


def guard_fit(seed: list[dict[str, Any]], fitted: list[dict[str, Any]],
              observations: list[dict[str, Any]], final_combo: int,
              max_pair_seconds: float = .5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose a fitted event set without worsening a final-combo seed.

    A complete fit wins when it improves the weighted sampled-count loss. If
    it regresses a seed that already has the target combo, the function
    greedily retains only individually helpful timing edits and nearby
    zero-sum promotion/demotion exchanges. Geometry and note paths stay from
    the fitted output; only the fit-owned fields are rolled back.
    """
    data = _samples(observations)
    if data is None:
        return fitted, {"state": "no-reliable-counter-samples"}
    times, counts, weights = data
    seed_count = sum(len(_atoms(e)) for e in seed)
    fit_count = sum(len(_atoms(e)) for e in fitted)
    seed_curve = _curve(seed, times)
    fit_curve = _curve(fitted, times)
    seed_loss = _loss(seed_curve, counts, weights)
    fit_loss = _loss(fit_curve, counts, weights)
    seed_exact = float(np.mean(seed_curve == counts))
    fit_exact = float(np.mean(fit_curve == counts))

    # A lower sampled error is a direct, data-backed improvement. Do not
    # constrain a fresh fit just because its visual seed has a weak combo.
    if (fit_count == int(final_combo) and _contact_safe(seed, fitted)
            and (seed_count != int(final_combo) or fit_loss <= seed_loss + 1e-6)):
        return fitted, {
            "state": "kept-improving-fit", "seedJudgements": seed_count,
            "fitJudgements": fit_count, "seedWeightedLoss": round(seed_loss, 4),
            "fitWeightedLoss": round(fit_loss, 4), "seedExactFraction": round(seed_exact, 5),
            "fitExactFraction": round(fit_exact, 5), "acceptedEdits": "all",
        }

    # Preserve candidate ordering; new candidates are treated as fake in the
    # seed so that a fitted promotion must earn its place from the video.
    base = copy.deepcopy(seed)
    if len(base) < len(fitted):
        for event in fitted[len(base):]:
            phantom = copy.deepcopy(event)
            phantom["isFake"] = True
            base.append(phantom)
    elif len(base) > len(fitted):
        base = base[:len(fitted)]
    if sum(len(_atoms(e)) for e in base) != int(final_combo):
        # There is no count-preserving prior to refine. Keep the best complete
        # fit and report the score so the caller can still audit it.
        return fitted, {
            "state": "kept-fit-no-complete-seed", "seedJudgements": seed_count,
            "fitJudgements": fit_count, "seedWeightedLoss": round(seed_loss, 4),
            "fitWeightedLoss": round(fit_loss, 4), "seedExactFraction": round(seed_exact, 5),
            "fitExactFraction": round(fit_exact, 5), "acceptedEdits": "all",
        }

    current = _curve(base, times)
    current_loss = _loss(current, counts, weights)
    changes: list[dict[str, Any]] = []
    for index, (old, new) in enumerate(zip(base, fitted)):
        changed = (bool(old.get("isFake")) != bool(new.get("isFake"))
                   or abs(float(old.get("time", 0)) - float(new.get("time", 0))) > 1e-5
                   or abs(float(old.get("duration", 0)) - float(new.get("duration", 0))) > 1e-5)
        if not changed:
            continue
        if old.get("isFake") and not new.get("isFake") and not _supported_promotion(new):
            continue
        # Strong source-contact measurements outrank counter-frame timing
        # jitter. They may still change note selection, but not drift far from
        # the visible contact point.
        if not old.get("isFake") and not new.get("isFake"):
            contact = old.get("sourceContactTime")
            strong = (float(old.get("ringContactSupport", 0) or 0) >= .75
                      and float(old.get("ringHeadSupport", 0) or 0) >= .75)
            if (strong and contact is not None
                    and abs(float(new["time"]) - float(contact)) > .025
                    and abs(float(new["time"]) - float(old["time"])) > .025):
                continue
        old_atoms = _atoms(old)
        new_atoms = _atoms(new)
        delta = _event_vector(new, times) - _event_vector(old, times)
        changes.append({"index": index, "event": new, "delta": delta,
                        "net": len(new_atoms) - len(old_atoms),
                        "when": min(float(old.get("time", 0)), float(new.get("time", 0)))})

    applied: set[int] = set()
    accepted = 0
    # Each accepted edit must lower independent source-counter error, and each
    # exchange retains the exact end-of-chart judgement count.
    for _ in range(min(200, len(changes))):
        best: tuple[float, list[dict[str, Any]]] | None = None
        remaining = [c for c in changes if c["index"] not in applied]
        for change in remaining:
            if change["net"] != 0:
                continue
            loss = _loss(current + change["delta"], counts, weights)
            if loss < current_loss - 1e-6 and (best is None or loss < best[0]):
                best = (loss, [change])
        positive = [c for c in remaining if c["net"] > 0]
        negative = [c for c in remaining if c["net"] < 0]
        for add in positive:
            for remove in negative:
                if add["net"] + remove["net"] != 0:
                    continue
                if abs(add["when"] - remove["when"]) > max_pair_seconds:
                    continue
                loss = _loss(current + add["delta"] + remove["delta"], counts, weights)
                if loss < current_loss - 1e-6 and (best is None or loss < best[0]):
                    best = (loss, [add, remove])
        if best is None:
            break
        current_loss, group = best
        for change in group:
            applied.add(change["index"])
            accepted += 1
            current += change["delta"]

    result = copy.deepcopy(fitted)
    fit_fields = set(_FIT_FIELDS)
    for i, event in enumerate(result):
        if i in applied:
            continue
        old = base[i]
        for key in fit_fields:
            if key in old:
                event[key] = copy.deepcopy(old[key])
            else:
                event.pop(key, None)
    if sum(len(_atoms(e)) for e in result) != int(final_combo):
        return fitted, {
            "state": "kept-fit-guard-count-mismatch", "seedJudgements": seed_count,
            "fitJudgements": fit_count, "seedWeightedLoss": round(seed_loss, 4),
            "fitWeightedLoss": round(fit_loss, 4), "acceptedEdits": accepted,
        }
    result_curve = _curve(result, times)
    result_loss = _loss(result_curve, counts, weights)
    return result, {
        "state": "conservative-counter-merge", "seedJudgements": seed_count,
        "fitJudgements": fit_count, "resultJudgements": int(final_combo),
        "seedWeightedLoss": round(seed_loss, 4), "fitWeightedLoss": round(fit_loss, 4),
        "resultWeightedLoss": round(result_loss, 4), "seedExactFraction": round(seed_exact, 5),
        "fitExactFraction": round(fit_exact, 5),
        "resultExactFraction": round(float(np.mean(result_curve == counts)), 5),
        "acceptedEdits": accepted,
    }
