"""Recover short source tracks discarded by the main detector's length gate."""
import json,math
from pathlib import Path
import numpy as np


def attach(events,root):
    tracks=json.loads((Path(root)/'tracks-v2.json').read_text())
    indexed={}
    for tr in tracks:
        s=np.asarray(tr['samples'],float)
        if len(s)<2:continue
        for sec in range(int(s[0,0]),int(s[-1,0])+1):indexed.setdefault(sec,[]).append(s)
    recovered=0
    for e in events:
        if e.get('path') or e.get('evidence')!='video-counter-and-fresh-source-hit-ring':continue
        t=e['time'];target=np.array([e['x'],e['judgeY']*720]);options=[]
        for s in indexed.get(int(t),[])+indexed.get(int(t)-1,[]):
            p=s[(s[:,0]>=t-.65)&(s[:,0]<=t+.015)]
            if len(p)<2 or p[-1,0]<t-.15 or p[0,0]>=t-.045:continue
            dt=p[-1,0]-p[0,0]
            velocity=(p[-1,1:3]-p[0,1:3])*2/max(.01,dt)
            if not 150<np.linalg.norm(velocity)<2400:continue
            prediction=p[-1,1:3]*2+velocity*(t-p[-1,0])
            miss=np.linalg.norm(prediction-target)
            if miss>32:continue
            # The track is source evidence even if too short for initial scan.
            options.append((miss-len(p)*1.2,p,velocity))
        if not options:continue
        _,p,v=min(options,key=lambda q:q[0])
        e['path']=[[float(q[0]),float(q[1]*2),float(q[2]*2)] for q in p]
        e['frames']=len(p);e['type']=int(np.median(p[:,5])>=.5)
        e['speed']=round(float(np.linalg.norm(v)),3)
        e['approachEvidence']='short-source-track-recovered-at-proven-hit'
        recovered+=1
    return {'shortObservedApproachesRecovered':recovered,
            'effectOnlyWithoutApproach':sum(not e.get('isFake') and not e.get('path') for e in events)}
