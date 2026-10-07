"""Measure uncropped capsules, bridging drag-shaped holes inside a hold body.

All inputs are video observations. A detector crop edge is never a hold tail.
"""
import json,copy
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage


def refine(video, root, events):
    index = {}
    measurements = {}
    separated_discs={}
    for i, e in enumerate(events):
        p = np.asarray(e.get('path', []), float)
        if e.get('duration', 0) <= 0 or len(p) < 2 or abs(e.get('rotation', 90)-90) > 8:
            continue
        finish = min(e['time']+2.6, max(p[-1, 0]+.15, e['time']+e['duration']+.2))
        for f in range(max(0, round(p[0, 0]*30)), round(finish*30)+1):
            t = f/30
            x = float(np.interp(t, p[:, 0], p[:, 1]))
            y = float(np.interp(t, p[:, 0], p[:, 2]))
            index.setdefault(f, []).append((i, x, y))
    cap = cv2.VideoCapture(str(video))
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        for i, x, y in index.get(f, []):
            x0, x1 = max(0, round(x)-28), min(1280, round(x)+29)
            crop = im[:, x0:x1].astype(float)
            # Capsule interiors stay bright blue even over a neutral white rail.
            bright = (crop[:, :, 0] > 235) & (crop.min(2) > 175)
            support = bright.sum(1)
            occupied = support >= 13
            # A drag overlaid on a hold can cut a 20–40 px hole in its interior.
            closed=occupied.copy()
            gaps,n_gaps=ndimage.label(~occupied)
            for k in range(1,n_gaps+1):
                ys=np.flatnonzero(gaps==k)
                above=support[max(0,ys[0]-8):ys[0]] if len(ys) else []
                below=support[ys[-1]+1:min(720,ys[-1]+9)] if len(ys) else []
                if 0<len(ys)<=43 and ys[0]>0 and ys[-1]<719 and len(above)>=8 and len(below)>=8 and np.median(above)>20 and np.median(below)>20 and (support[ys]>=4).mean()>.5:
                    closed[ys]=True
            e=events[i]
            # Neutral line beams can connect two independent round taps into
            # one component. A true hold has a bright interior between its caps.
            old_tail=np.asarray(e.get('tailPath',[]),float)
            if f/30<e['time']-.08 and len(old_tail)>1:
                ty=float(np.interp(f/30,old_tail[:,0],old_tail[:,2]))
                lo_y,hi_y=round(ty+12),round(y-12)
                if 112<lo_y<hi_y<565 and 40<hi_y-lo_y<180:
                    middle=support[lo_y:hi_y]
                    if (middle>=13).mean()<.65 and support[max(0,round(ty)-5):round(ty)+6].max()>=13 and support[round(y)-5:min(720,round(y)+6)].max()>=13:
                        separated_discs.setdefault(i,[]).append([f/30,y-ty])
            labels, n = ndimage.label(closed)
            candidates = []
            discs=[]
            for k in range(1, n+1):
                ys = np.flatnonzero(labels == k)
                solid = ys[occupied[ys]]
                if len(solid) < 28:
                    continue
                top, bottom = int(solid[0]), int(solid[-1])+1
                width = float(np.median(support[solid]))
                if 23<=bottom-top<=53 and width>23:
                    discs.append((top+bottom)/2)
                width = float(np.clip(width, 25, 44))
                head = bottom-width/2
                if abs(head-y) > 115 or bottom-top < 36 or bottom-top-width<15:
                    continue
                candidates.append((abs(head-y), head, top+width/2, top < 3, width))
            if candidates:
                _, head, tail, clipped, width = min(candidates)
                measurements.setdefault(i, []).append([f/30, x, head, tail, clipped, width])
            if f/30<e['time']-.08:
                lower=[c for c in discs if abs(c-y)<25]
                if lower:
                    head=min(lower,key=lambda c:abs(c-y))
                    upper=[c for c in discs if 45<head-c<190]
                    if upper:separated_discs.setdefault(i,[]).append([f/30,head-max(upper)])
        f += 1
    cap.release()
    changes, endpoints = 0, 0
    splits=[]
    for i,evidence in separated_discs.items():
        if len(evidence)<3:
            continue
        e=events[i]
        gap=float(np.median([q[1] for q in evidence]))
        if not 45<gap<190:
            continue
        upper=copy.deepcopy(e)
        delta=gap/max(300.,e['speed'])
        upper.update(time=round(e['time']+delta,5),sourceTime=round(e['time']+delta,5),duration=0.,
                     path=copy.deepcopy(e['tailPath']),tailPath=[],frames=e['frames'],type=0,
                     evidence='separate-discs-over-neutral-line',holdSplitEvidence='two-bright-circles-with-neutral-gap')
        e.update(duration=0.,tailPath=[],type=0,evidence='separate-discs-over-neutral-line',
                 holdSplitEvidence='two-bright-circles-with-neutral-gap')
        splits.append(upper)
    for i, raw in measurements.items():
        e = events[i]
        if e['duration']<=0:
            continue
        s = np.asarray(raw, float)
        if len(s) < 3:
            continue
        gaps=np.flatnonzero(np.diff(s[:,0])>.12)
        if len(gaps):s=s[:gaps[0]+1]
        valid = s[s[:, 4] == 0]
        heads, tails = [], []
        for t, x, head, tail, clipped, width in s:
            if clipped:
                later = valid[(valid[:, 0] > t) & (valid[:, 0] < t+.9)]
                if len(later) >= 3:
                    later = later[:8]
                    velocity = np.median(np.diff(later[:, 3])/np.maximum(.01, np.diff(later[:, 0])))
                    if 100 < velocity < 2200:
                        tail = min(-25., float(later[0, 3]+velocity*(t-later[0, 0])))
                    else:
                        tail = -60.
                else:
                    tail = -60.
                endpoints += 1
            heads.append([float(t), float(x), float(head)])
            tails.append([float(t), float(x), float(tail)])
        hp = np.asarray(heads)
        # Keep the measured approach and use the game's stationary held head.
        p = np.asarray(e['path'], float)
        for q in p:
            if hp[0, 0] <= q[0] <= hp[-1, 0]:
                q[2] = float(np.interp(q[0], hp[:, 0], hp[:, 2]))
        e['path'] = heads
        e['bodyPath'] = heads
        e['tailPath'] = tails
        e['bodyEvidence'] = 'uncropped-bright-capsule-with-overlap-holes'
        e['holdBodyWidth'] = round(float(np.median(s[:, 5])), 3)
        # A long pre-hit body does not imply a long hold: some video holds
        # sharply change flow at contact. Estimate the end from the last
        # stationary capsule and the tail's observed collapse/disappearance.
        pinned = s[(s[:, 0] >= e['time']-.035) & (abs(s[:, 2]-e['judgeY']*720) < 35)]
        if len(pinned):
            # Do not bridge a disappearing hold to a later note on the lane.
            part = [pinned[0]]
            for q in pinned[1:]:
                if q[0]-part[-1][0] > .1:
                    break
                part.append(q)
            part = np.asarray(part)
            last = part[-1]
            finish = float(last[0]+1/30)
            if len(part) >= 2 and not last[4]:
                near = part[-min(5, len(part)):]
                v = float(np.median(np.diff(near[:, 3])/np.maximum(.01, np.diff(near[:, 0]))))
                if 100 < v < 2500:
                    finish = float(last[0]+max(0, e['judgeY']*720-last[3])/v)
                    finish = min(finish, last[0]+.09)
            proposed = max(.03, finish-e['time'])
            if abs(proposed-e['duration']) < 1.5:
                e['previousDuration'] = e['duration']
                e['duration'] = round(proposed, 5)
                e['holdEndEvidence'] = 'stationary-capsule-tail-and-disappearance'
        changes += 1
    events.extend(splits)
    rejected=[]
    for e in list(events):
        p=np.asarray(e.get('path',[]),float)
        if e.get('duration',0)>0 and e.get('frames',0)<6 and len(p)<3 and len(p) and p[:,2].min()>650:
            e['evidence']='non-note-particle-fragment';e['isFake']=True;rejected.append(e);events.remove(e)
    (Path(root)/'rejected-particle-fragments.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2))
    report = {'refittedCapsules': changes, 'offscreenTailSamples': endpoints,
              'measuredFrames': sum(map(len, measurements.values())),
              'pairedTapsSplitFromBeamComponents':len(splits),'rejectedParticleFragments':len(rejected)}
    (Path(root)/'hold-refine-v4.json').write_text(json.dumps(report, indent=2))
    print('Uncropped hold refinement:', report, flush=True)
    return report


def consolidate(events):
    """A cropped/occluded capsule may have several detector track IDs."""
    merged=0
    for a in sorted(list(events),key=lambda e:e.get('path',[[e['time']]])[0][0]):
        if a not in events or a.get('duration',0)<=0 or len(a.get('bodyPath',[]))<4:continue
        for b in list(events):
            if b is a or b.get('duration',0)<=0 or len(b.get('bodyPath',[]))<4:continue
            ap=np.asarray(a['bodyPath']);bp=np.asarray(b['bodyPath'])
            if bp[0,0]<ap[0,0]-.001 or bp[0,0]>ap[-1,0]+.10:continue
            common=bp[(bp[:,0]>=ap[0,0])&(bp[:,0]<=ap[-1,0])]
            if len(common)<3:continue
            delta=np.hypot(common[:,1]-np.interp(common[:,0],ap[:,0],ap[:,1]),common[:,2]-np.interp(common[:,0],ap[:,0],ap[:,2]))
            if (delta<8).sum()<3 or (delta<8).mean()<.65:continue
            # Preserve one observed capsule; do not duplicate its head/tail
            # combo just because two thresholds split its component track.
            def union(p,q):
                d={round(v[0],5):v for v in p+q};return [d[t] for t in sorted(d)]
            a['path']=union(a['path'],b['path']);a['bodyPath']=union(a['bodyPath'],b['bodyPath'])
            a['tailPath']=union(a['tailPath'],b['tailPath'])
            a['duration']=max(a['time']+a['duration'],b['time']+b['duration'])-a['time']
            a['frames']+=b['frames'];a['path']=a['bodyPath'];a['mergedTracks']=a.get('mergedTracks',[a.get('track')])+[b.get('track')]
            a['isFake']=False;events.remove(b);merged+=1
    from hold_fit import fit_holds
    refitted=fit_holds(events)
    return {'mergedCapsuleFragments':merged,'postMeasurementContactsRefitted':refitted}
