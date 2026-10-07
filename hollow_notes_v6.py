"""Track individual hollow drops even when their outer borders touch.

Each closed inner contour remains a separate observation in a dense chain.
No source chart, counter-derived position, or hand-authored note is used.
"""
import json,math,hashlib,inspect
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

def detections(im,t):
 b,g,r=[im[:,:,k].astype(np.int16) for k in range(3)]
 mask=((b>170)&(g>110)&(r>130)&(b-g>10)).astype('uint8')*255
 cs,hi=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);out=[]
 if hi is None:return out
 for i,c in enumerate(cs):
  if hi[0,i,3]<0:continue
  x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c)
  if not(20<=min(w,h)<=36 and max(w,h)<=46 and 280<area<1100):continue
  p=cv2.approxPolyDP(cv2.convexHull(c),2,True).reshape(-1,2).astype(float)
  if len(p)<4:continue
  angles=[]
  for j,q in enumerate(p):
   a=p[j-1]-q;bb=p[(j+1)%len(p)]-q
   angles.append(math.degrees(math.acos(float(np.clip(a@bb/max(.01,np.linalg.norm(a)*np.linalg.norm(bb)),-1,1)))))
  j=int(np.argmin(angles));sharp=angles[j]
  if not 70<sharp<118:continue
  center=np.array([x+(w-1)/2,y+(h-1)/2]);tip=p[j];angle=math.degrees(math.atan2(*(tip-center)[::-1]))
  axis=(tip-center)/max(1,np.linalg.norm(tip-center));normal=np.array([-axis[1],axis[0]])
  ratio=np.ptp(p@axis)/max(1,np.ptp(p@normal))
  if ratio<1.06:continue
  # The bbox of the inner hole and the source sprite share their centre.
  out.append([t,*center,angle,(w+h)/2+10,sharp])
 return out

def _angle_gap(a,b):
 d=abs((float(a)-float(b)+180.)%360.-180.)
 return d

def scan(video,root):
 root=Path(root);cache=root/'hollow-notes-v6.json'
 calibration=root/'calibration.json'
 nominal=957.28
 if calibration.exists():
  try:nominal=float(json.loads(calibration.read_text()).get('nominalPixelsPerSecondAt1280',nominal))
  except (ValueError,TypeError):pass
 max_speed=max(2400.,nominal*2.7)
 sig=hashlib.sha256((inspect.getsource(detections)+inspect.getsource(_angle_gap)+inspect.getsource(scan)).encode()+str(Path(video).stat().st_size).encode()+str(round(nominal,3)).encode()).hexdigest()
 if cache.exists():
  old=json.loads(cache.read_text())
  if old.get('signature')==sig:return old['tracks']
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);f=0;tracks=[];active=[];cv2.setNumThreads(2)
 while True:
  ok,im=cap.read()
  if not ok:break
  t=f/fps;ds=detections(im,t);used=set();nxt=[]
  if active and ds:
   costs=np.full((len(active),len(ds)),1e6)
   for i,tr in enumerate(active):
    s=np.asarray(tr['samples']);last=s[-1];dt=t-last[0];v=np.array([0.,960.])
    if len(s)>=2:v=np.clip((last[1:3]-s[-2,1:3])/max(.01,last[0]-s[-2,0]),-max_speed,max_speed)
    for j,d in enumerate(ds):
     dist=np.linalg.norm(np.array(d[1:3])-last[1:3]-v*dt)
     step=np.linalg.norm(np.array(d[1:3])-last[1:3])/max(.001,dt)
     lane_step=abs(float(d[2])-float(last[2]))
     angle_gap=_angle_gap(d[3],last[3])
     # Fast-flow drops can move laterally by more than 60 pixels between
     # captured frames. The former nearly-static gate split them into
     # single-frame decorations before their matching judge target was seen.
     # Use calibrated flow with shape/orientation and cross-lane checks.
     gate=26.+dt*max_speed
     if (dist<gate and step<max_speed*1.15 and
         lane_step<54.+dt*480. and angle_gap<58. and
         abs(d[4]-last[4])<12):
      costs[i,j]=dist+abs(d[4]-last[4])*.4+angle_gap*.12
   ii,jj=linear_sum_assignment(costs)
   for i,j in zip(ii,jj):
    if costs[i,j]>1000:continue
    tr=active[i];tr['samples'].append(ds[j]);nxt.append(tr);used.add(j)
  for j,d in enumerate(ds):
   if j not in used:
    tr={'id':800000+len(tracks),'samples':[d]};tracks.append(tr);nxt.append(tr)
  for tr in active:
   if not any(tr is q for q in nxt) and t-tr['samples'][-1][0]<.07:nxt.append(tr)
  active=nxt;f+=1
  if f%900==0:print('V6 independent hollow contours',f,flush=True)
 cap.release();good=[]
 for tr in tracks:
  s=np.asarray(tr['samples'])
  if np.median(s[:,5])>105:continue
  if len(s)<2 or np.ptp(s[:,1:3],axis=0).max()<24:
   # Video speed changes can expose a whole drop for a single frame. Preserve
   # such observations above the judgement area as unscored native sprites.
   if np.max(s[:,2])<520 and np.median(s[:,5])<102:
    tr.update(speed=960.,visualOnly=True);good.append(tr)
   continue
  v=np.linalg.norm(np.diff(s[:,1:3],axis=0),axis=1)/np.diff(s[:,0]);speed=float(np.median(v))
  if not 130<speed<20000:continue
  tr['speed']=speed;good.append(tr)
 cache.write_text(json.dumps({'signature':sig,'tracks':good},separators=(',',':')));return good

def merge(video,root,events,targets):
 tracks=scan(video,root);heads={};replaced=0;added=0;used=set()
 for tr in targets:
  for q in tr['samples']:
   if q['radius']<28 and q['alpha']>.2:heads.setdefault(round(q['time']*30),[]).append((q['x'],q['y'],tr['id']))
 for tr in sorted(tracks,key=lambda q:len(q['samples']),reverse=True):
  s=np.asarray(tr['samples']);p=s[:,:3];matches=[]
  if tr.get('visualOnly'):
   # Do not add a snapshot where a longer source track already draws it.
   shared=False
   for e in events:
    ep=np.asarray(e.get('path',[]),float)
    if len(ep)<2:continue
    common=p[(p[:,0]>=ep[0,0])&(p[:,0]<=ep[-1,0])]
    if len(common) and np.linalg.norm(common[:,1:]-np.column_stack([np.interp(common[:,0],ep[:,0],ep[:,k]) for k in (1,2)]),axis=1).min()<18:shared=True;break
   if shared:continue
   t=p[-1,0]+1/30;x,y=p[-1,1:]
   e={'time':float(t),'sourceTime':float(t),'x':float(x),'sourceX':float(x),'judgeY':float(y/720),'duration':0,'isFake':True,'visualOnly':True,'confidence':'observed-sprite-only','evidence':'source-hollow-single-frame-sprite','rotation':90.}
   events.append(e);used.add(len(events)-1);added+=1
   e.update(type=1,path=p.tolist(),frames=len(p),speed=tr['speed'],hollowTrack=tr['id'])
   e['orientationPath']=[[float(tm),float(a)] for tm,a in zip(s[:,0],np.degrees(np.unwrap(np.radians(s[:,3]))))]
   e['noteScale']=float(np.clip(np.median(s[:,4])/(2000*.0223*(335/185)*.54),.6,1.3));continue
  for i,e in enumerate(events):
   if i in used or e['duration']>0:continue
   ep=np.asarray(e.get('path',[]),float)
   if len(ep)<2 or p[-1,0]<ep[0,0] or p[0,0]>ep[-1,0]:continue
   common=p[(p[:,0]>=ep[0,0])&(p[:,0]<=ep[-1,0])]
   if len(common)<3:continue
   ds=np.linalg.norm(common[:,1:]-np.column_stack([np.interp(common[:,0],ep[:,0],ep[:,k]) for k in (1,2)]),axis=1)
   if (ds<15).mean()>.7:matches.append((float(np.median(ds))-min(10,len(common))*.1,i))
  # A longer contour path can expose a target crossing after an earlier
  # detector has already emitted a weak event at the sprite's first sighting.
  # Fit the contact on the observed trajectory in both cases; otherwise that
  # early guess consumes a combo before the moving note reaches its target.
  candidates=[]
  for j in range(max(1,len(p)-8),len(p)):
   first=max(0,j-3);dt=max(.01,p[j,0]-p[first,0]);v=(p[j,1:]-p[first,1:])/dt;v2=v@v
   if v2<130**2:continue
   for f in range(round(p[j,0]*30),round((p[j,0]+.18)*30)+1):
    for x,y,ident in heads.get(f,[]):
     d=np.array([x,y])-p[j,1:];jump=float(d@v/v2);miss=float(np.linalg.norm(d-v*jump))
     if -.04<jump<.18 and abs(p[j,0]+jump-f/30)<.05 and miss<15:candidates.append((miss+abs(jump)*10,p[j,0]+jump,x,y,ident))
  contact=min(candidates) if candidates else None
  if matches:
   _,i=min(matches);e=events[i];used.add(i);replaced+=1
   if contact is not None and e['duration']==0:
    _,t,x,y,ident=contact
    old_time=float(e['time'])
    e.update(time=round(float(t),5),sourceTime=round(float(t),5),sourceContactTime=round(float(t),5),
             counterPinnedTime=round(float(t),5),fitBaseTime=round(float(t),5),x=float(x),sourceX=float(x),
             judgeY=float(y/720),targetTrack=ident,preTargetContactTime=old_time,
             targetContactEvidence='flow-linked-hollow-intersection')
  else:
   if contact is not None:_,t,x,y,ident=contact
   else:t=p[-1,0]+1/30;x,y=p[-1,1:];ident=None
   # Replace an effect-only guess when this approach reaches its exact hit.
   guesses=[];recent=p[-min(4,len(p)):];v=(recent[-1,1:]-recent[0,1:])/max(.01,recent[-1,0]-recent[0,0])
   for i,q in enumerate(events):
    if i in used or q['duration'] or q.get('frames',0)>3 or q.get('evidence')!='video-counter-and-fresh-source-hit-ring':continue
    dt=q['time']-p[-1,0];miss=np.linalg.norm(p[-1,1:]+v*dt-np.array([q['x'],q['judgeY']*720]))
    if -.02<dt<.12 and miss<24:guesses.append((miss,i))
   if guesses:
    _,i=min(guesses);e=events[i];used.add(i);replaced+=1
    if contact is not None:
     _,t,x,y,ident=contact
     e.update(time=round(float(t),5),sourceTime=round(float(t),5),sourceContactTime=round(float(t),5),
              counterPinnedTime=round(float(t),5),fitBaseTime=round(float(t),5),x=float(x),sourceX=float(x),
              judgeY=float(y/720),targetTrack=ident,targetContactEvidence='flow-linked-hollow-intersection')
   else:
    e={'time':float(t),'sourceTime':float(t),'x':float(x),'sourceX':float(x),'judgeY':float(y/720),'duration':0,'isFake':True,'targetTrack':ident,'confidence':'medium','evidence':'independent-faint-source-approach','rotation':90.}
    if contact is not None:
     e.update(sourceContactTime=round(float(t),5),counterPinnedTime=round(float(t),5),
              targetContactEvidence='flow-linked-hollow-intersection',hollowCandidateDeferred=True)
    events.append(e);used.add(len(events)-1);added+=1
  e.update(type=1,path=p.tolist(),frames=len(p),speed=tr['speed'],hollowTrack=tr['id'],approachEvidence='independent-full-resolution-inner-drop-contours')
  if contact is not None:
   e['targetContactEvidence']='flow-linked-hollow-intersection'
  angles=np.degrees(np.unwrap(np.radians(s[:,3])));e['orientationPath']=[[float(tm),float(a)] for tm,a in zip(s[:,0],angles)]
  e['noteScale']=float(np.clip(np.median(s[:,4])/(2000*.0223*(335/185)*.54),.6,1.3))
 # Different detector passes can follow the same fast hollow approach. Merge
 # those observations after target fitting so they cannot become two playable
 # notes at one contact while another lane's real note remains hidden.
 hollow=[]
 for e in events:
  if e.get('type')!=1 or e.get('duration',0)>0:continue
  p=np.asarray(e.get('path',[]),float)
  if len(p)<2 or not (e.get('hollowTrack') is not None or e.get('targetTrack') is not None or
                      'hollow' in str(e.get('evidence','')) or e.get('approachEvidence')):continue
  hollow.append((e,p))
 removed=set();merged_duplicates=0
 def duplicate(a,pa,b,pb):
  if a.get('targetTrack') is not None and a.get('targetTrack')==b.get('targetTrack') and abs(a['time']-b['time'])<.055:
   return math.hypot(float(a['x'])-float(b['x']),float(a['judgeY']*720-b['judgeY']*720))<44
  lo=max(pa[0,0],pb[0,0]);hi=min(pa[-1,0],pb[-1,0])
  if hi-lo>=.045:
   ts=np.linspace(lo,hi,4)
   aa=np.column_stack([np.interp(ts,pa[:,0],pa[:,k]) for k in (1,2)])
   bb=np.column_stack([np.interp(ts,pb[:,0],pb[:,k]) for k in (1,2)])
   if np.median(np.linalg.norm(aa-bb,axis=1))<20 and abs(a['time']-b['time'])<.065:return True
  return (a.get('targetTrack') is not None or b.get('targetTrack') is not None) and \
         abs(float(a['time'])-float(b['time']))<.045 and \
         math.hypot(float(a['x'])-float(b['x']),float(a['judgeY']*720-b['judgeY']*720))<36
 def rank(e):
  return (8*int(e.get('targetTrack') is not None)+2*float(e.get('ringContactSupport',0) or 0)+
          float(e.get('ringHeadSupport',0) or 0)+min(3.,len(e.get('path',[]))*.12))
 for i,(a,pa) in enumerate(hollow):
  if id(a) in removed:continue
  for b,pb in hollow[i+1:]:
   if id(b) in removed or not duplicate(a,pa,b,pb):continue
   keeper,other=(a,b) if rank(a)>=rank(b) else (b,a)
   for key in ('ringContactSupport','ringHeadSupport','headHitSupport'):
    if other.get(key) is not None:keeper[key]=max(float(keeper.get(key,0) or 0),float(other[key]))
   keeper['mergedHollowObservation']=True
   removed.add(id(other));merged_duplicates+=1
 if removed:events[:]=[e for e in events if id(e) not in removed]
 return {'measuredHollowTracks':len(tracks),'hollowApproachesReplaced':replaced,
         'hollowCandidatesAdded':added,'duplicateHollowApproachesMerged':merged_duplicates}

def restore_unrepresented_visual_tracks(events,tracks,min_frames=8,min_span=.24):
 """Keep sustained source-visible hollow tracks when later fitting drops them.

 This is a visual-only recovery pass: it never creates a judgement or changes
 combo counts. A source contour track is considered represented only when an
 existing event follows the same path over enough shared frames. Nearby,
 parallel notes remain separate because matching uses full trajectory distance.
 """
 restored=[]
 for tr in tracks:
  samples=np.asarray(tr.get('samples',[]),float)
  if samples.ndim!=2 or len(samples)<min_frames or samples.shape[1]<6:continue
  path=samples[:,:3]
  if path[-1,0]-path[0,0]<min_span or float(np.median(samples[:,5]))>98:continue
  represented=False
  for e in events:
   other=np.asarray(e.get('path',[]),float)
   if other.ndim!=2 or len(other)<2 or other.shape[1]<3:continue
   lo=max(path[0,0],other[0,0]);hi=min(path[-1,0],other[-1,0])
   if hi-lo<.07:continue
   common=path[(path[:,0]>=lo)&(path[:,0]<=hi)]
   if len(common)<4:continue
   predicted=np.column_stack([np.interp(common[:,0],other[:,0],other[:,k]) for k in (1,2)])
   distances=np.linalg.norm(common[:,1:3]-predicted,axis=1)
   if np.median(distances)<18 and (distances<22).mean()>=.7:
    represented=True;break
  if represented:continue
  angles=np.degrees(np.unwrap(np.radians(samples[:,3])))
  scale=float(np.clip(np.median(samples[:,4])/(2000*.0223*(335/185)*.54),.6,1.3))
  event={'time':float(path[-1,0]+1/30),'sourceTime':float(path[-1,0]+1/30),
         'x':float(path[-1,1]),'sourceX':float(path[-1,1]),
         'judgeY':float(path[-1,2]/720),'type':1,'duration':0,
         'isFake':True,'visualOnly':True,'confidence':'observed-sprite-only',
         'evidence':'source-hollow-unrepresented-visual-track','rotation':90.,
         'path':path.tolist(),'frames':len(path),'speed':float(tr.get('speed',960)),
         'hollowTrack':tr.get('id'),'orientationPath':[[float(q),float(a)] for q,a in zip(samples[:,0],angles)],
         'noteScale':scale}
  events.append(event);restored.append(tr.get('id'))
 return {'restoredVisualTracks':len(restored),'trackIds':restored}
