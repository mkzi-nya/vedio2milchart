"""Use visible approaches, fresh hit particles and per-frame combo jointly."""
import json, math
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def _trajectory_conflict(a, b):
    """Return true when two detections are cuts of one visible approach.

    The full-resolution pass intentionally keeps weak, long approaches so a
    note hidden by particles is not lost.  Those approaches can overlap the
    short target pass for the same note, however.  Compare the measured paths
    in their common time span instead of comparing only heads: the latter
    misses duplicates whose heads were found on different frames.
    """
    if a.get('duration', 0) > 0 or b.get('duration', 0) > 0:
        return False
    pa = np.asarray(a.get('path', []), float)
    pb = np.asarray(b.get('path', []), float)
    if len(pa) < 4 or len(pb) < 4:
        return False
    lo = max(float(pa[0, 0]), float(pb[0, 0]))
    hi = min(float(pa[-1, 0]), float(pb[-1, 0]))
    if hi - lo < .085:
        return False
    samples = np.linspace(lo, hi, 5)
    da = np.column_stack([np.interp(samples, pa[:, 0], pa[:, k]) for k in (1, 2)])
    db = np.column_stack([np.interp(samples, pb[:, 0], pb[:, k]) for k in (1, 2)])
    distances = np.linalg.norm(da - db, axis=1)
    # A single crossing is insufficient evidence.  A duplicate follows the
    # same approach for most of the overlap and ends at the same judge line.
    return float(np.median(distances)) < 30. and float(np.max(distances)) < 58.


def _duplicate_approach(e, events):
    """Detect an unsupported long approach duplicated by a ring-backed one."""
    if e.get('ringContactSupport', 0) or e.get('counterRequired'):
        return False
    if e.get('evidence') not in ('full-screen-decorative-track',
                                 'visible-track-without-local-counter',
                                 'joint-counter-and-track-hypothesis') and not e.get('approachEvidence'):
        return False
    for other in events:
        if other is e or other.get('isFake'):
            continue
        if other.get('ringContactSupport', 0) < .60:
            continue
        if _trajectory_conflict(e, other):
            return True
    return False


def recover_approaches(events, targets, hitframes):
    heads = {}
    for tr in targets:
        for q in tr['samples']:
            if q['radius'] < 28 and q['alpha'] > .25:
                heads.setdefault(round(q['time']*30), []).append(q)
    recovered = 0
    for e in events:
        if e.get('evidence') not in ('full-screen-decorative-track', 'visible-track-without-local-counter', 'joint-counter-and-track-hypothesis') or not e.get('isFake'):
            continue
        p = np.asarray(e.get('path', []), float)
        if len(p) < 3 or e.get('duration', 0) > 0:
            continue
        recent = p[-min(5, len(p)):]
        dt = recent[-1, 0]-recent[0, 0]
        velocity = (recent[-1, 1:]-recent[0, 1:])/max(.01, dt)
        speed = np.linalg.norm(velocity)
        if not 300 < speed < 2400:
            continue
        predictions = recent[0, 1:]+(recent[:, 0]-recent[0, 0])[:, None]*velocity
        if np.max(np.linalg.norm(predictions-recent[:, 1:], axis=1)) > 25:
            continue
        last = p[-1]; options = []
        for f in range(round(last[0]*30), round((last[0]+.25)*30)+1):
            for h in heads.get(f, []):
                target = np.array([h['x'], h['y']])
                jump = float((target-last[1:]) @ velocity/(speed*speed))
                residual = np.linalg.norm(last[1:]+velocity*jump-target)
                if -.02 <= jump <= .23 and residual < 14 and abs(last[0]+jump-f/30) < .055:
                    options.append((residual+abs(jump)*15, last[0]+jump, h))
        if not options:
            continue
        _, t, h = min(options, key=lambda q: q[0])
        if any(other is not e and not other.get('isFake') and abs(other['time']-t) < .055
               and math.hypot(other['x']-h['x'], other['judgeY']*720-h['y']) < 35 for other in events):
            continue
        e.update(time=round(float(t), 5), sourceTime=round(float(t), 5), x=float(h['x']),
                 sourceX=float(h['x']), judgeY=float(h['y']/720), targetTrack=h.get('id'),
                 rotation=90., approachEvidence='occluded-approach-to-visible-target', isFake=False)
        e['path'] = p.tolist()+[[round(float(t), 5), float(h['x']), float(h['y'])]]
        recovered += 1
    # Prefer a fresh rise in local purple particles to an old nearby particle.
    for e in events:
        def energy(t):
            values = [power*math.exp(-((x-e['x'])**2+(y-e['judgeY']*720)**2)/(2*65**2))
                      for frame in hitframes if abs(frame['time']-t) < .04
                      for x, y, power in frame['centers']]
            return max(values, default=0.)
        before = max(energy(e['time']-.10), energy(e['time']-.17))
        after = max(energy(e['time']+.04), energy(e['time']+.10))
        e['headHitSupport'] = round(max(0., after-before*.65), 5)
    return recovered


def _independently_supported(e):
    return bool(
        float(e.get('ringContactSupport', 0) or 0) >= .55
        or float(e.get('ringHeadSupport', 0) or 0) >= .45
        or float(e.get('headHitSupport', 0) or 0) >= .30
        or (e.get('targetTrack') is not None and
            (e.get('sourceContactTime') is not None or e.get('approachEvidence')))
    )


def _weak_faint_candidate(e):
    return (e.get('evidence') == 'faint-hollow-moving-decoration'
            and not _independently_supported(e)
            and e.get('targetTrack') is None
            and not e.get('approachEvidence'))


def _counter_transition_windows(observations):
    """Return observed combo-rise windows and their timing tolerance."""
    rows = []
    for q in observations or []:
        try:
            t, combo = float(q['time']), int(q['combo'])
            error, margin = float(q.get('error', 1)), float(q.get('margin', 0))
        except (KeyError, TypeError, ValueError):
            continue
        if not ((error < .085 and margin > .025) or (error < .045 and margin > .020)):
            continue
        rows.append((t, combo))
    rows.sort()
    transitions = []
    for previous, current in zip(rows, rows[1:]):
        gap = current[0] - previous[0]
        if gap <= 0 or current[1] <= previous[1]:
            continue
        tolerance = min(.16, max(.06, gap * .5 + .03))
        transitions.append((previous[0], current[0], tolerance, current[1] - previous[1]))
    return transitions


def fit(events, observations, final_combo=1875, time_limit=45):
    for e in events:
        e.setdefault('fitBaseTime',e['time']);e.setdefault('fitBaseDuration',e['duration'])
        e['time']=e['fitBaseTime'];e['duration']=e['fitBaseDuration']
    # Reliable persistent source counts are cumulative constraints at that
    # exact video frame. Keep both ends of long unchanged plateaus.
    clean = [q for q in observations if (q['error'] < .085 and q['margin'] > .025) or (q['error']<.045 and q['margin']>.020)]
    groups = []
    for q in clean:
        if not groups or q['combo'] != groups[-1][-1]['combo']:
            groups.append([])
        groups[-1].append(q)
    anchors = [(0., 0)]
    for part in groups:
        if part[0]['time'] > anchors[-1][0] and part[0]['combo'] >= anchors[-1][1]:
            anchors.append((part[0]['time'], part[0]['combo']))
        if part[-1]['time']-part[0]['time'] > .05:
            anchors.append((part[-1]['time'], part[-1]['combo']))
    if anchors[-1][0] < 141.:
        anchors.append((141., final_combo))
    n, m = len(events), len(anchors)-1
    # Build this once; doing pairwise path interpolation inside every MILP
    # option would make the fitting cost grow quadratically with candidates.
    duplicate_approaches = {id(e) for e in events if _duplicate_approach(e, events)}
    rows, cols, values, options, costs = [], [], [], [], []
    times = np.asarray([q[0] for q in anchors[1:]])
    expected = np.diff([q[1] for q in anchors])
    transition_windows = _counter_transition_windows(observations)
    quiet = np.flatnonzero(expected==0)
    quiet_rows = {int(interval):n+m+1+i for i,interval in enumerate(quiet)}
    local_row_start=n+m+1+len(quiet)
    for j, e in enumerate(events):
        if e.get('visualOnly'):continue
        blocked = (e.get('evidence') == 'decorative-track-hypothesis' and e.get('ringHeadSupport',0)<.25) or (e.get('evidence') == 'full-screen-decorative-track' and not e.get('approachEvidence') and not e.get('ringContactSupport'))
        # Do not use distance from the nominal screen centre as evidence of a
        # fake.  The judgement line itself rotates and translates, so a real
        # hit can be far from (x=640,y=592), and the violet ring is frequently
        # occluded by the previous burst.  Track continuity and the local
        # counter decide these candidates; ring support remains a quality term.
        if (e.get('evidence') in ('dynamic-target-intersection','occluded-independent-pass',
                                  'separate-discs-over-neutral-line','independent-full-resolution-capsule')
                and e.get('frames',0)>=3):
            blocked = bool(e.get('visualOnly'))
        # A target detector can emit a second short segment on the same
        # moving track after the real hit. When that duplicate has no local
        # ring contact, prefer the supported segment and keep it decorative.
        if not e.get('ringContactSupport') and e.get('targetTrack') is not None:
            blocked |= any(other is not e and other.get('targetTrack') == e.get('targetTrack')
                           and abs(other.get('time', 0)-e.get('time', 0)) < .11
                           and other.get('ringContactSupport')
                           for other in events)
        # A long full-screen approach and a short ring-backed target segment
        # can be two cuts of one note. Keep the physically supported segment
        # and make the duplicate decorative so it cannot consume combo.
        blocked |= id(e) in duplicate_approaches
        pp=np.asarray(e.get('path',[]),float)
        blocked |= e.get('duration',0)>.2 and e.get('frames',0)<8 and e.get('ringHeadSupport',0)<.15 and len(pp)>1 and np.ptp(pp[:,1:],axis=0).max()<80
        # Newly discovered target-intersecting tracks are deferred until the
        # local counter pass asks for them. Existing note tracks that gained a
        # better measured contact remain available in the first fit; their
        # contact time is pinned to the observed target intersection.
        if e.get('hollowCandidateDeferred'):
            blocked = not e.get('counterRequired')
        if e.get('evidence') == 'independent-faint-source-approach':
            direct = (float(e.get('ringContactSupport',0) or 0)>=.55 or
                      float(e.get('ringHeadSupport',0) or 0)>=.45 or
                      e.get('targetTrack') is not None)
            blocked |= not direct and not e.get('counterRequired')
        if e.get('counterRequired'):
            blocked = False
        if blocked:
            continue
        # A pinned source-ring contact is independent physical evidence and
        # must override the earlier counter-only pin.  The old branch order
        # made counterPinnedContact ineffective whenever counterPinnedTime
        # was already present, so a strong hit could stay several frames off
        # its measured contact despite being explicitly protected above.
        strong_contact_pin = (
            e.get('counterPinnedContact')
            and e.get('sourceContactTime') is not None
            and float(e.get('ringContactSupport', 0) or 0) >= .75
            and float(e.get('ringHeadSupport', 0) or 0) >= .75
        )
        if strong_contact_pin:
            shifts = (round(float(e['sourceContactTime'])-e['time'], 5),)
        elif e.get('counterPinnedTime') is not None:
            shifts = (round(float(e['counterPinnedTime'])-e['time'], 5),)
        else:
            shifts = (-1/30,0.,1/30) if e.get('sourceContactTime') else (-2/30,-1/30,0.,1/30,2/30)

        # The HUD is sampled at 30 fps, while the actual judgement can occur
        # between two readable counter frames.  A candidate just outside an
        # interval boundary used to be forced into the next interval, which
        # made one local combo increment disappear and shifted all later
        # assignments.  Add only boundary snaps within 1.5 source frames;
        # the event remains tied to its measured trajectory and the fit still
        # chooses exactly one interval for each judgement atom.
        def boundary_shifts(base_time, current):
            values = {round(float(s), 5) for s in current}
            for start, end in zip([a[0] for a in anchors[:-1]], [a[0] for a in anchors[1:]]):
                if end - start <= .006:
                    continue
                # Video time is reconstructed from the encoded frame clock,
                # while the hit ring and the HUD counter can each lag by one
                # frame. Let the local assignment cross a boundary by at most
                # 2.5 frames when that is needed to satisfy the observed combo
                # interval. The source-contact penalty below still favors the
                # measured hit time, so this does not become a global time warp.
                if start - .084 <= base_time < start:
                    values.add(round(float(start + .001 - base_time), 5))
                elif end < base_time <= end + .084:
                    values.add(round(float(end - .001 - base_time), 5))
            return tuple(sorted(values))
        # A strong source-ring contact is a physical timing observation.  Keep
        # its measured contact when a counter boundary would otherwise pull
        # it into the neighbouring interval.  Counter-only repair events are
        # also pinned to the interval that created them; snapping them back
        # across that boundary would undo the repair before it is exported.
        pinned = ((e.get('counterPinnedContact') and e.get('sourceContactTime') is not None)
                  or e.get('counterRequired') or e.get('counterPinnedTime') is not None)
        if not pinned:
            shifts = boundary_shifts(float(e['time']), shifts)
        tail_base = (-1/30, 0., 1/30) if e.get('sourceContactTime') else (-2/30, -1/30, 0., 1/30, 2/30)
        if e.get('counterPinnedContact') and e.get('sourceContactTime') is not None:
            tail_base = (round(float(e['sourceContactTime']) - e['time'], 5),)
        tail_shifts = boundary_shifts(float(e['time'] + e['duration']), tail_base) if e['duration'] > 0 else (0.,)
        from itertools import product
        for shift,tail_shift in product(shifts,tail_shifts):
            t = round(max(0., e['time']+shift),5); end = round(e['time']+e['duration']+tail_shift,5)
            if e['duration']>0 and end<t+.015:continue
            if _weak_faint_candidate(e):
                if not any(abs(t - end_time) <= tolerance
                           for _, end_time, tolerance, _ in transition_windows):
                    continue
            col = len(options)
            options.append((j, shift,tail_shift)); rows.append(j); cols.append(col); values.append(1)
            # No source combo increment means no real judgement. Quiet source
            # intervals are hard constraints, including a held tail.
            for hit in ([t, end] if e['duration'] > 0 else [t]):
                interval = int(np.searchsorted(times, hit-1e-6))
                if interval < m:
                    rows.append(n+interval); cols.append(col); values.append(1)
                    rows.append(local_row_start+interval);cols.append(col);values.append(1)
                    if interval in quiet_rows:
                        rows.append(quiet_rows[interval]);cols.append(col);values.append(1)
            rows.append(n+m); cols.append(col); values.append(1+int(e['duration'] > 0))
            quality = np.log1p(e.get('frames', 0))*.4+bool(e.get('targetTrack') is not None)*1.5
            # Prefer an actual expanding contact burst at the target over a
            # long visible approach alone. These measurements are independent
            # of note shape and remain useful when a note is partly hidden.
            quality += min(6., float(e.get('ringContactSupport', 0) or 0)*6.)
            quality += min(5., float(e.get('ringHeadSupport', 0) or 0)*6.)
            quality += min(7., float(e.get('headHitSupport', 0) or 0)*10.)
            quality += bool(e.get('approachEvidence'))*.8
            quality -= 3.5*int(not e.get('ringContactSupport'))
            quality -= 4.*int(e.get('frames',0)==0)
            quality += .25*bool(not e.get('isFake'))
            quality += 80.*int(e.get('counterRequired',False))
            contact_penalty = 0.
            contact = e.get('sourceContactTime')
            if contact is not None and e.get('ringContactSupport', 0) >= .60:
                # Source contact is estimated from the hit ring and the
                # measured approach.  It is a stronger timing observation
                # than a one-frame OCR boundary; retain it unless the local
                # counter constraints make the candidate impossible.
                contact_penalty = abs(t - float(contact)) * 85.
            costs.append(-quality+abs(shift)*25+abs(tail_shift)*18+contact_penalty)
    # Missing judgements stay explicit. Never turn a fake into a real note
    # solely because the final total would otherwise be short.
    option_count = len(options)
    for i in range(m):
        for sign in (-1, 1):
            col = len(costs); rows.append(n+i); cols.append(col); values.append(sign)
            # Penalize cumulative counter error, so a missing event in one
            # interval cannot silently shift every later counter by that amount.
            if i+1<m:
                rows.append(n+i+1);cols.append(col);values.append(-sign)
            # A reliable source-frame counter is stronger evidence than a
            # marginally better visual candidate. Keep residuals available
            # for genuinely missing detections, but make them much more
            # expensive so the optimizer first satisfies local HUD counts.
            # Source-frame OCR anchors are local evidence.  A residual here
            # changes the phase of every later combo, so it must be much more
            # expensive than selecting a lower-confidence candidate.
            costs.append(200000.)
    # Local residuals matter independently of the cumulative total. Without
    # these rows a missing source event could be compensated by turning a
    # later decorative note real. Such compensation matches the final combo
    # while creating audible judgements at the wrong moment.
    for i in range(m):
        for sign in (-1,1):
            col=len(costs);rows.append(local_row_start+i);cols.append(col);values.append(sign)
            # Keep the independently measured frame interval exact whenever
            # a ring-backed candidate can satisfy it.
            costs.append(300000.)
    matrix = coo_matrix((values, (rows, cols)), shape=(local_row_start+m, len(costs))).tocsc()
    # A repair is evidence, not a second independent observation. Several
    # detector passes can point at the same source hit, and a hold can place
    # its head and tail in different counter intervals. Making every repair
    # an event-level hard lower bound can therefore make a valid fit
    # infeasible. Repairs remain strongly preferred by the objective below;
    # the exact per-interval combo constraints decide whether they are used.
    lower = np.r_[np.zeros(n), expected, 0, np.zeros(len(quiet)),expected]
    upper = np.r_[np.ones(n), expected, final_combo, np.zeros(len(quiet)),expected]
    # Solve the source-frame allocation in two stages.  The old implementation
    # always exposed residual variables, so it could satisfy the final combo by
    # putting an extra hit in one interval and compensating with a missing hit
    # much later.  That preserves the total while moving actual judgements to
    # the wrong moment.  Prefer a strict feasible allocation first; residuals
    # are only a last resort for a genuinely missing or unreadable detection.
    strict_ub = np.r_[np.ones(option_count), np.zeros(len(costs)-option_count)]
    result = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0, strict_ub),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': time_limit, 'mip_rel_gap': .005})
    used_residuals = False
    if result.x is None:
        ub = np.r_[np.ones(option_count), np.full(len(costs)-option_count, final_combo)]
        result = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0, ub),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': time_limit, 'mip_rel_gap': .01})
        used_residuals = True
    if result.x is None:
        return {'state': 'unresolved', 'message': result.message}
    flags = np.rint(result.x[:option_count]).astype(int)
    for e in events:
        e['isFake'] = True
    shifted = 0
    for flag, (j, shift, tail_shift) in zip(flags, options):
        if not flag:
            continue
        e = events[j]; e['isFake'] = False
        old_end=e['time']+e['duration']
        if shift:
            e['previousHeadTime'] = e['time']; e['time'] = round(max(0., e['time']+shift), 5)
            e['timingAdjustment'] = round(shift, 5); shifted += 1
        if e['duration']>0:
            e['duration']=round(old_end+tail_shift-e['time'],5)
            e['tailTimingAdjustment']=round(tail_shift,5)
        e['judgementEvidence'] = 'visible-approach-source-hit-ring-and-video-frame-counter'
    cumulative = []
    for time, combo in anchors[1:]:
        actual = sum(int(e['time'] <= time+1e-6)+int(e['duration']>0 and e['time']+e['duration'] <= time+1e-6) for e in events if not e['isFake'])
        cumulative.append({'time': time, 'videoCombo': combo, 'reconstructedCombo': actual, 'difference': actual-combo})
    return {'state': 'fitted-video-frame-counter', 'anchors': m, 'shiftedHeads': shifted,
            'solverStatus': int(result.status), 'exactAnchors': sum(q['difference']==0 for q in cumulative),
            'maxCounterDifference': max(abs(q['difference']) for q in cumulative),
            'hardQuietIntervals':len(quiet), 'strictLocalAllocation': not used_residuals,
            'cumulative': cumulative}
