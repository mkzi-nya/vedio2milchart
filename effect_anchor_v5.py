"""Source hit rings, not bright note outlines, determine judgement anchors.

The violet effect has a radius larger than a note and surrounds empty space.
Its circular support is checked before accepting a Hough proposal. No chart
files are used. All stored positions are source video pixels.
"""
import json, math, hashlib
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def scan(video, root):
    root = Path(root)
    cache = root/'source-hit-rings-v5.json'
    import inspect
    signature = hashlib.sha256(inspect.getsource(scan).encode()+str(Path(video).resolve()).encode()+str(Path(video).stat().st_size).encode()).hexdigest()
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get('signature') == signature:
            return old
    cv2.setNumThreads(2)
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []; angles = np.arange(96)*2*np.pi/96
    co, si = np.cos(angles), np.sin(angles)
    f = 0
    while True:
        ok, im = cap.read()
        if not ok: break
        im = cv2.resize(im, (640,360), interpolation=cv2.INTER_AREA)
        b,g,r = [im[:,:,k].astype(float) for k in range(3)]
        mask = ((b-g>25)&(b-r>10)&(g>55)&(b>120)).astype('uint8')*255
        mask[:52] = 0
        wide=cv2.dilate(mask,np.ones((3,3),np.uint8))
        found = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 14,
                                 param1=60, param2=14, minRadius=17, maxRadius=57)
        rings = []
        for x,y,rad in ([] if found is None else found[0]):
            # Require a genuinely circular arc rather than scattered petals,
            # a capsule or two neighbouring note outlines.
            values = []
            for delta in (-1.5,-.5,.5,1.5):
                xx = np.rint(x+(rad+delta)*co).astype(int)
                yy = np.rint(y+(rad+delta)*si).astype(int)
                inside = (xx>=0)&(xx<640)&(yy>=52)&(yy<360)
                v = np.zeros(96);v[inside]=mask[yy[inside],xx[inside]]/255
                values.append(v)
            visible = (y+rad*si>=52)&(y+rad*si<360)&(x+rad*co>=0)&(x+rad*co<640)
            if visible.sum()<60: continue
            sectors = np.max(values,axis=0)
            coverage = float(sectors[visible].mean())
            # At least six octants must have support. Crossing particles only
            # occupy a few angular sectors even when Hough votes are high.
            octants = sectors.reshape(8,12).mean(1)
            if coverage<.57 or (octants>.35).sum()<6: continue
            small_support=0.
            for rr in range(18,32,2):
                xx=np.rint(x+rr*co).astype(int);yy=np.rint(y+rr*si).astype(int)
                valid=(xx>=0)&(xx<640)&(yy>=52)&(yy<360)
                if valid.sum()<64:continue
                small_support=max(small_support,float((wide[yy[valid],xx[valid]]>0).mean()))
            rings.append({'x':round(float(x*2),2),'y':round(float(y*2),2),
                          'radius':round(float(rad*2),2),'support':round(coverage,4),
                          'smallSupport':round(small_support,4)})
        frames.append({'time':round(f/fps,6),'rings':rings})
        f += 1
        if f%900==0: print('source hit rings:',f,'frames',flush=True)
    cap.release()
    result = {'signature':signature,'fps':fps,'samples':frames,
              'rings':sum(len(q['rings']) for q in frames)}
    cache.write_text(json.dumps(result,separators=(',',':')))
    return result


def fit(events, targets, source,retime=True):
    fps = source['fps']; frames = source['samples']
    circles = {}
    for tr in targets:
        if tr['samples'][0]['radius']>28: continue
        for q in tr['samples']:
            circles.setdefault(round(q['time']*fps),[]).append(q)
    located, moving, unsupported = 0,0,0
    for e in events:
        if e.get('visualOnly'):continue
        p = np.asarray(e.get('path',[]),float)
        t = e['time']; dur = e.get('duration',0); x=e['x']; y=e['judgeY']*720
        if len(p)<2:
            # Effect-only hypotheses still need source ring evidence. Their
            # synthetic approach is never evidence for making them real.
            p=np.array([[t-.1,x,y],[t+.1,x,y]])
        # Contact must be on the observed approach, not a remotely located
        # circle intersected by an extrapolation of the wrong capsule cap.
        candidates=[]
        def approach_contact(rx,ry):
            """A delayed nearby ring must lie on this particular note's path."""
            found=[];point=np.array([rx,ry])
            if len(e.get('path',[]))<3:return found
            for u,v in zip(p,p[1:]):
                dt=v[0]-u[0];delta=v[1:]-u[1:];length2=float(delta@delta)
                if not 0<dt<.15 or length2<16:continue
                frac=float((point-u[1:])@delta/length2);ct=float(u[0]+frac*dt)
                miss=float(np.linalg.norm(u[1:]+frac*delta-point))
                if -.2<frac<1.2 and miss<22 and abs(ct-t)<(.3 if retime else .052):found.append((miss,ct))
            if p[-1,0]<t+.06:
                last=p[-min(3,len(p)):];dt=last[-1,0]-last[0,0]
                velocity=(last[-1,1:]-last[0,1:])/max(.01,dt);speed2=float(velocity@velocity)
                if speed2>200**2:
                    jump=float((point-last[-1,1:])@velocity/speed2);ct=float(last[-1,0]+jump)
                    miss=float(np.linalg.norm(last[-1,1:]+velocity*jump-point))
                    if -.02<jump<.22 and miss<22 and abs(ct-t)<(.3 if retime else .052):found.append((miss,ct))
            return found
        unresolved=e.get('evidence') in ('full-screen-decorative-track','independent-faint-source-approach','faint-hollow-moving-decoration')
        lo=max(0,round((t-.16)*fps));hi=min(len(frames),round((t+(.36 if unresolved else .23))*fps)+1)
        for f in range(lo,hi):
            tm=frames[f]['time']
            hx=float(np.interp(tm,p[:,0],p[:,1]));hy=float(np.interp(tm,p[:,0],p[:,2]))
            if unresolved and p[-1,0]<tm<p[-1,0]+.28 and len(p)>3:
                recent=p[-min(5,len(p)):];delta=recent[-1,0]-recent[0,0]
                velocity=(recent[-1,1:]-recent[0,1:])/max(.01,delta)
                if 200<np.linalg.norm(velocity)<2400:
                    hx,hy=p[-1,1:]+velocity*(tm-p[-1,0])
            for q in frames[f]['rings']:
                distance=math.hypot(q['x']-x,q['y']-y)
                approach=math.hypot(q['x']-hx,q['y']-hy)
                direct_target_ring = (
                    unresolved and e.get('targetTrack') is not None
                    and len(e.get('path',[])) >= 5
                    and abs(q['x']-x) <= 30 and abs(q['y']-y) <= 30
                    and abs(tm-t-.06) <= .075
                    and q.get('support',0) >= .60
                    and q.get('smallSupport',0) >= .50
                )
                if (dur<=0 and len(e.get('path',[]))>=3
                        and not approach_contact(q['x'],q['y'])
                        and not direct_target_ring):
                    continue
                if unresolved:
                    if (abs(q['x']-x)>180 or distance>320
                            or (approach>60 and not direct_target_ring)):
                        continue
                    previous=max((old['smallSupport'] for pf in frames[max(0,f-4):f]
                                  for old in pf['rings'] if math.hypot(old['x']-q['x'],old['y']-q['y'])<20),default=0.)
                    if q['smallSupport']-previous*.95<.25:continue
                    score=distance*.1+approach*.5+abs(tm-t-.06)*60-q['support']*12
                else:
                    if abs(q['x']-x)>42 or distance>100 or approach>110:continue
                    score=distance*.35+approach*.15+abs(tm-t-.025)*100-q['support']*12
                candidates.append((score,tm,q))
        if not candidates:
            e['ringHeadSupport']=0.
            if retime:e.pop('ringContactSupport',None)
            unsupported+=1;continue
        _, tm, ring = min(candidates,key=lambda q:q[0])
        rx,ry=ring['x'],ring['y']
        # Snap the effect centre to an observed small circle only when it agrees
        # with the ring. This removes Hough's few-pixel centre quantisation.
        near=[q for f in range(max(0,round(tm*fps)-2),round(tm*fps)+3)
              for q in circles.get(f,[]) if math.hypot(q['x']-rx,q['y']-ry)<12]
        if near:
            q=min(near,key=lambda q:math.hypot(q['x']-rx,q['y']-ry)+abs(q['time']-t)*40)
            rx,ry=float(q['x']),float(q['y'])
        if retime and len(e.get('path',[]))>=3 and not e.get('capsuleContactTime'):
            pp=np.asarray(e.get('bodyPath',e['path']) if dur>0 else e['path'],float)
            contacts=[]
            for u,v in zip(pp,pp[1:]):
                if v[0]-u[0]>.15 or v[0]<=u[0]:continue
                delta=v[1:]-u[1:];length2=float(delta@delta)
                if length2<16:continue
                frac=float((np.array([rx,ry])-u[1:])@delta/length2)
                ct=float(u[0]+frac*(v[0]-u[0]));miss=np.linalg.norm(u[1:]+frac*delta-[rx,ry])
                if -.2<frac<1.2 and miss<20 and abs(ct-t)<(.3 if unresolved else .18):
                    contacts.append((abs(ct-t)+miss/400,ct))
            if not contacts and pp[-1,0]<t+.06:
                last=pp[-min(5,len(pp)):];dt=last[-1,0]-last[0,0]
                velocity=(last[-1,1:]-last[0,1:])/max(.01,dt);speed2=float(velocity@velocity)
                if speed2>200**2:
                    jump=float((np.array([rx,ry])-last[-1,1:])@velocity/speed2)
                    miss=np.linalg.norm(last[-1,1:]+velocity*jump-[rx,ry])
                    if -.02<jump<(.28 if unresolved else .17) and miss<22 and abs(last[-1,0]+jump-t)<(.3 if unresolved else .18):
                        contacts.append((0.,float(last[-1,0]+jump)))
            if contacts:
                ct=min(contacts)[1];old_end=t+dur
                if dur<=0 or old_end>ct+.015:
                    e['sourceContactTime']=round(ct,5);e['time']=round(ct,5)
                    if dur>0:e['duration']=round(old_end-e['time'],5)
                    t=e['time'];dur=e['duration']
        if e.get('capsuleContactTime') is not None and dur>0:
            # The ring is delayed, and its centre can move with the held cap.
            # Transport the measured effect back to contact instead of
            # mistaking later sideways motion for a new head intersection.
            body=np.asarray(e.get('bodyPath',e.get('path',[])),float)
            if len(body)>1:
                physical=np.array([np.interp(tm,body[:,0],body[:,k]) for k in (1,2)])
                contact=np.array([np.interp(t,body[:,0],body[:,k]) for k in (1,2)])
                delta=np.clip(np.array([rx,ry])-physical,-5,5)
                rx,ry=(contact+delta).tolist()
                e['sourceContactTime']=e['capsuleContactTime']
        def small(t0,t1):
            return max((q['smallSupport'] for f in range(max(0,round(t0*fps)),min(len(frames),round(t1*fps)+1))
                        for q in frames[f]['rings'] if math.hypot(q['x']-rx,q['y']-ry)<20),default=0.)
        before=small(t-.13,t-.035);after=small(t-.02,t+.10)
        e['ringHeadSupport']=round(max(0,after-before*.95),4)
        e['ringContactSupport']=ring['support']
        e.setdefault('originalJudgePoint',[x,y])
        e['x']=rx;e['judgeY']=ry/720
        anchors=[[t,rx,ry]]
        if dur>0:
            previous=np.array([rx,ry]); velocity=np.zeros(2);last=t
            body=np.asarray(e.get('bodyPath',e.get('path',[])),float)
            for f in range(round(t*fps)+1,min(len(frames),round((t+dur)*fps)+1)):
                ft=frames[f]['time'];dt=ft-last;prediction=previous+velocity*dt
                choices=[]
                for q in frames[f]['rings']:
                    if q['radius']>68:continue
                    pos=np.array([q['x'],q['y']]);d=np.linalg.norm(pos-prediction)
                    if len(body)>1 and body[0,0]-.04<=ft<=body[-1,0]+.04:
                        physical=np.array([np.interp(ft,body[:,0],body[:,k]) for k in (1,2)])
                        if np.linalg.norm(pos-physical)>25:continue
                    if d<35+dt*300:choices.append((d-q['support']*8,pos))
                if choices:
                    pos=min(choices,key=lambda q:q[0])[1]
                    cc=[q for q in circles.get(f,[]) if math.hypot(q['x']-pos[0],q['y']-pos[1])<10]
                    if cc:
                        q=min(cc,key=lambda q:math.hypot(q['x']-pos[0],q['y']-pos[1]));pos=np.array([q['x'],q['y']])
                    velocity=np.clip((pos-previous)/max(.02,dt),-1100,1100)*.6+velocity*.4
                    previous=pos;last=ft;anchors.append([ft,float(pos[0]),float(pos[1])])
            anchors.append([t+dur,float(previous[0]),float(previous[1])])
            if np.ptp(np.array(anchors)[:,1:],axis=0).max()>8:moving+=1
        e['anchorPath']=anchors;e['anchorEvidence']='source-violet-hit-ring-and-matching-circle';located+=1
    return {'effectAnchors':located,'movingHoldAnchors':moving,'withoutLocalRing':unsupported}


def supplement_hits(events,allocation,source):
    """Add only a new source ring in an interval with missing source combo.

    Existing head/tail contacts, old expanding rings and quiet source spans
    cannot create a repair. The final solver may still reject a hypothesis.
    """
    if not allocation.get('cumulative'):return []
    fps=source['fps'];frames=source['samples'];intervals=[];prev=(0.,0,0)
    # The circular part of the effect appears after the actual contact. Infer
    # this delay from visible note/target intersections in this same video.
    lags=[]
    for e in events:
        ct=e.get('sourceContactTime')
        if ct is None or e.get('ringHeadSupport',0)<.5:continue
        onset=[f['time'] for f in frames[max(0,round((ct-.05)*fps)):round((ct+.16)*fps)]
               if any(math.hypot(q['x']-e['x'],q['y']-e['judgeY']*720)<20 and q['smallSupport']>.72 for q in f['rings'])]
        if onset and .02<min(onset)-ct<.13:lags.append(min(onset)-ct)
    lag=float(np.median(lags)) if len(lags)>=20 else 2/fps
    source['inferredContactToRingDelay']=lag
    for q in allocation['cumulative']:
        start,video,actual=prev;missing=(q['videoCombo']-video)-(q['reconstructedCombo']-actual)
        if missing>0:intervals.append((start,q['time'],missing))
        prev=(q['time'],q['videoCombo'],q['reconstructedCombo'])
    found=[]
    spent={}
    def already_repaired(tm, x, y, include_events=False):
        """Collapse detector duplicates without merging simultaneous notes."""
        pool=found
        if include_events:
            pool=pool+[e for e in events if e.get('repairSource') or e.get('counterRequired')]
        return any(abs(float(other.get('time',0))-tm)<.025 and
                   math.hypot(float(other.get('x',0))-x,
                              float(other.get('judgeY',.82))*720-y)<32
                   for other in pool)
    # First recover candidates that were already tracked but rejected by the
    # visual-only filters.  A local combo deficit is stronger evidence than a
    # missing ring (the ring is routinely hidden by the preceding burst).
    # Reuse the measured path and only snap the judgement time to this exact
    # counter interval.  This keeps position/type/hold information coupled to
    # the observation instead of synthesizing a bare note.
    for start, end, budget in intervals:
        if budget <= 0:
            continue
        nearby=[]
        for e in events:
            if not e.get('isFake') or e.get('duration',0)>0:
                continue
            if e.get('counterRequired'):
                continue
            if not (start-.055 < e.get('time',0) <= end+.055):
                continue
            evidence=e.get('evidence','')
            if evidence in ('source-hollow-single-frame-sprite','faint-hollow-moving-decoration'):
                continue
            path_len=len(e.get('path',[])); track_bonus=1.5 if e.get('targetTrack') is not None else 0.
            ring_bonus=min(2.,float(e.get('ringHeadSupport',0)))
            continuity=min(2.,math.log1p(e.get('frames',0))*.35)
            # Tracks with no local path and no target identity are still
            # allowed only when a source ring later confirms them below.
            if path_len<3 and e.get('targetTrack') is None and ring_bonus<.35:
                continue
            score=track_bonus+ring_bonus+continuity
            nearby.append((score,e))
        for _,e in sorted(nearby,key=lambda q:q[0],reverse=True):
            if budget<=0:break
            t=float(e['time'])
            snapped=min(max(t,start+.001),end-.001)
            if any(abs(other.get('time',0)-snapped)<.018 and
                   math.hypot(other.get('x',0)-e.get('x',0),
                              (other.get('judgeY',.82)-e.get('judgeY',.82))*720)<28
                   for other in found):
                continue
            e['time']=round(snapped,5);e['sourceTime']=round(snapped,5)
            e['fitBaseTime']=e['time'];e['fitBaseDuration']=e.get('duration',0)
            e['counterPinnedTime']=e['time'];e['counterRequired']=True
            e['repairCounterInterval']=[start,end]
            e['repairSource']='video-counter-track-repair'
            e['evidence']='video-counter-track-repair';e['isFake']=False
            found.append(e);budget-=1;spent[(start,end)]=spent.get((start,end),0)+1

    for start,end,budget in intervals:
        budget=max(0,budget-spent.get((start,end),0))
        candidates=[]
        for f in range(max(1,round(start*fps)),min(len(frames)-1,round(end*fps)+2)):
            tm=frames[f]['time']-lag
            if not start<tm<=end:continue
            for q in frames[f]['rings']:
                # A fresh inner burst can be coaxial with an older expanding
                # ring. Hough reports the outer radius; the independent inner
                # profile and its onset still prove the new hit.
                if q['smallSupport']<.50 or q['support']<.55:continue
                before=max((p['smallSupport'] for prevf in range(max(0,f-4),f)
                            for p in frames[prevf]['rings'] if math.hypot(p['x']-q['x'],p['y']-q['y'])<23),default=0.)
                fresh=q['smallSupport']-before*.95
                if fresh<.18:continue
                if not any(math.hypot(p['x']-q['x'],p['y']-q['y'])<15 and p['support']>.6 for p in frames[f+1]['rings']):continue
                if not already_repaired(tm,q['x'],q['y'],include_events=True):
                    candidates.append((fresh*q['support'],tm,q))
        for strength,tm,q in sorted(candidates,reverse=True,key=lambda p:p[0]):
            contacts=[(e['time'],e['x'],e['judgeY']*720) for e in events if not e.get('isFake')]
            contacts += [(e['time']+e['duration'],e['anchorPath'][-1][1] if e.get('anchorPath') else e['x'],e['anchorPath'][-1][2] if e.get('anchorPath') else e['judgeY']*720) for e in events if not e.get('isFake') and e.get('duration',0)>0]
            if any(abs(ct-tm)<.095 and math.hypot(cx-q['x'],cy-q['y'])<38 for ct,cx,cy in contacts):continue
            if already_repaired(tm,q['x'],q['y']):continue
            possible=[e for e in events if e.get('isFake') and e.get('duration',0)==0 and abs(e['time']-tm)<.16 and abs(e['x']-q['x'])<35 and abs(e['judgeY']*720-q['y'])<80]
            if possible:
                e=min(possible,key=lambda e:abs(e['time']-tm)*100+abs(e['x']-q['x']))
                # Keep its measured approach; only the source contact is fitted.
                e['time']=round(tm,5);e['evidence']='video-counter-and-fresh-source-hit-ring'
                e['sourceEffectDelay']=lag
                e['fitBaseTime']=e['time'];e['fitBaseDuration']=e['duration']
                e['ringHeadSupport']=strength;e['ringContactSupport']=q['support'];e['isFake']=False
                e['x']=q['x'];e['judgeY']=q['y']/720;e['anchorPath']=[[tm,q['x'],q['y']],[tm+e['duration'],q['x'],q['y']]]
                found.append(e)
            else:
                e={'time':round(tm,5),'sourceTime':round(tm,5),'x':q['x'],'sourceX':q['x'],'judgeY':q['y']/720,
                   'duration':0.,'type':0,'frames':0,'speed':957.28,'isFake':False,
                   'confidence':'inferred','evidence':'video-counter-and-fresh-source-hit-ring',
                   'ringHeadSupport':strength,'ringContactSupport':q['support'],
                   'path':[],'anchorPath':[[tm,q['x'],q['y']]],
                   'repairCounterInterval':[start,end],'repairSourceFrame':f,
                   'repairSource':'video-counter-and-fresh-source-hit-ring'}
                e['sourceEffectDelay']=lag
                events.append(e);found.append(e)
            budget-=1
            if budget<=0:break

        # If the effect detector was occluded, reuse the nearest remaining
        # tracked hypothesis as a last local repair.  It must still have a
        # multi-frame path or a target identity; single-frame hollow sprites
        # remain excluded because they are the dominant false-positive class.
        if budget>0:
            nearby=[]
            for e in events:
                if not e.get('isFake') or e.get('duration',0)>0:
                    continue
                if e.get('counterRequired'):
                    continue
                if not (start-.09 < e.get('time',0) <= end+.09):continue
                if e.get('evidence') in ('source-hollow-single-frame-sprite','faint-hollow-moving-decoration'):continue
                if e.get('frames',0)<3 and e.get('targetTrack') is None:continue
                distance=0 if start<=e['time']<=end else min(abs(e['time']-start),abs(e['time']-end))
                nearby.append((distance,e))
            for _,e in sorted(nearby,key=lambda q:q[0]):
                if budget<=0:break
                snapped=min(max(float(e['time']),start+.001),end-.001)
                if any(abs(other.get('time',0)-snapped)<.018 and
                       math.hypot(other.get('x',0)-e.get('x',0),
                                  (other.get('judgeY',.82)-e.get('judgeY',.82))*720)<28
                       for other in found):continue
                e['time']=round(snapped,5);e['sourceTime']=e['time'];e['fitBaseTime']=e['time']
                e['counterPinnedTime']=e['time'];e['counterRequired']=True
                e['repairCounterInterval']=[start,end]
                e['repairSource']='video-counter-boundary-repair'
                e['evidence']='video-counter-boundary-repair';e['isFake']=False
                found.append(e);budget-=1
    return found
