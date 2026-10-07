"""Full-resolution capsule tracking, including tilted and rotating long notes.

Store both cap centres, measured width and source-supported body lifetime.
The bounding-box width of a tilted hold is not its physical thickness.
"""
import json,math,hashlib,copy
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import median_filter

def capsule_rotations(heads,tails,snap_tolerance_degrees=.4):
 """Estimate each hold axis from its measured cap pair without flattening real tilt."""
 heads=np.asarray(heads,float);tails=np.asarray(tails,float)
 vector=tails[:,1:]-heads[:,1:]
 angles=np.degrees(np.unwrap(np.arctan2(-vector[:,1],vector[:,0])))
 angles=median_filter(angles,size=3,mode='nearest')
 angles[np.abs(angles-90)<snap_tolerance_degrees]=90
 return angles

def refine_occluded_hold_orientations(events):
 """Repair sustained angle conflicts using detected capsule cap-pair geometry."""
 corrected=[]
 for e in events:
  if (not e.get('duration') or e.get('bodyEvidence') not in {
       'uncropped-bright-capsule-with-overlap-holes',
       'full-resolution-blue-capsule-cap-pair'} or not e.get('holdRotationPath')):continue
  heads=np.asarray(e.get('bodyPath') or e.get('path') or [],float)
  tails=np.asarray(e.get('tailPath') or [],float)
  old=np.asarray(e['holdRotationPath'],float)
  if len(heads)<4 or len(heads)!=len(tails) or np.max(np.abs(heads[:,0]-tails[:,0]))>1e-4:continue
  lengths=np.linalg.norm(tails[:,1:]-heads[:,1:],axis=1)
  # A cap centre may sit just beyond the image edge while the capsule body
  # still supplies a stable axis.  Exclude far-offscreen extrapolations, but
  # keep a modest border margin so the first/last visible frames can correct
  # stale rotations instead of inheriting them.
  edge_margin=720*.22
  visible=((heads[:,2]>=-edge_margin)&(heads[:,2]<=720+edge_margin)
           &(tails[:,2]>=-edge_margin)&(tails[:,2]<=720+edge_margin)
           &(lengths>=80)&(lengths<=800))
  if int(visible.sum())<4:continue
  hp=heads[visible];tp=tails[visible];measured=capsule_rotations(hp,tp)
  existing=np.interp(hp[:,0],old[:,0],old[:,1])
  disagreement=np.abs((measured-existing+90)%180-90)
  run=0;longest=0;previous=None
  for tm,error in zip(hp[:,0],disagreement):
   if error>8 and (previous is None or tm-previous<=.05):run+=1
   elif error>8:run=1
   else:run=0
   longest=max(longest,run);previous=float(tm)
  # Three adjacent source frames are enough to reject single-frame edge noise
  # while still recovering a brief, clipped entrance immediately before the
  # longer visible orientation run.
  if longest<3:continue
  replacements={float(tm):float(angle) for tm,angle in zip(hp[:,0],measured)}
  combined={float(tm):float(angle) for tm,angle in old}
  combined.update(replacements)
  e['holdRotationPath']=[[tm,combined[tm]] for tm in sorted(combined)]
  e['rotation']=float(np.interp(float(e['time']),[q[0] for q in e['holdRotationPath']],
                               [q[1] for q in e['holdRotationPath']]))
  corrected.append({'time':float(e['time']),'track':e.get('track'),'samples':int(visible.sum()),
                    'maxDisagreement':float(np.max(disagreement)),'consecutive':int(longest)})
 return {'corrected':len(corrected),'events':corrected}

def capsules(im,t):
 b,g,r=[im[:,:,i].astype(float) for i in range(3)]
 mask=((b>225)&(g>170)&(r>175)&(b-g>8)).astype('uint8')*255
 coverage=(b>220)&(g>135)&(r>125)&(b-g>6)
 # Keep the playable area above the HUD where possible. HUD text is neutral;
 # chromatic interiors distinguish blue capsules even when they pass behind it.
 mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);ds=[]
 for c in contours:
  area=cv2.contourArea(c)
  if area<850:continue
  rect=cv2.minAreaRect(c);corners=cv2.boxPoints(rect);lens=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=1);j=int(np.argmin(lens));short=float(lens[j]);long=float(max(lens))
  if not 24<short<140 or long-short<24 or area/(short*long)<.66:continue
  a=(corners[j]+corners[(j+1)%4])/2;b=(corners[(j+2)%4]+corners[(j+3)%4])/2;d=(b-a)/max(1,np.linalg.norm(b-a));a+=d*short/2;b-=d*short/2
  head,tail=(a,b) if a[1]>b[1] else (b,a)
  # A neutral rail joining round notes has no continuous blue interior.
  pp=head+(tail-head)*np.linspace(.12,.88,15)[:,None];xx=pp[:,0].round().astype(int).clip(0,1279);yy=pp[:,1].round().astype(int).clip(0,719)
  if (mask[yy,xx]>0).mean()<.8:continue
  x0,y0,w,h=cv2.boundingRect(c);clipped=(x0<=1 or y0<=1 or x0+w>=1279 or y0+h>=719)
  # Clean segmentation splits a body under its violet hit particles. Probe a
  # broad transverse strip rather than one axis pixel: continuous blue sides
  # establish the body even when particles darken its centre.
  axis=(head-tail)/max(1,np.linalg.norm(head-tail));normal=np.array([-axis[1],axis[0]])
  def extend_cap(point,direction):
   distances=np.arange(0,650,1.);offsets=np.linspace(-.32,.32,9)*short
   pts=point+distances[:,None,None]*direction+offsets[None,:,None]*normal
   xx=pts[:,:,0].round().astype(int);yy=pts[:,:,1].round().astype(int)
   valid=(xx>=0)&(xx<1280)&(yy>=0)&(yy<720)
   support=(coverage[yy.clip(0,719),xx.clip(0,1279)]&valid).mean(axis=1)>.65
   stop=len(support);gap=0
   for k,yes in enumerate(support):
    gap=0 if yes else gap+1
    if gap>=4:stop=k-3;break
   good=np.flatnonzero(support[:stop])
   if not len(good):return point
   extension=float(good[-1])-.43*short
   return point+direction*extension if extension>12 else point
  head=extend_cap(head,axis);tail=extend_cap(tail,-axis)
  q={'time':round(t,6),'head':head.astype(float),'tail':tail.astype(float),'width':short,'clipped':clipped}
  # Both clean fragments can now describe the same continuous physical body.
  if not any(np.linalg.norm(v['head']-head)<14 and np.linalg.norm(v['tail']-tail)<14 for v in ds):ds.append(q)
 return ds

def scan(video,root):
 root=Path(root);cache=root/'capsules-v6.json';import inspect
 root=Path(root);cache=root/'capsules-v6.json';sig=hashlib.sha256((inspect.getsource(capsules)+inspect.getsource(scan)+inspect.getsource(serial)).encode()+str(Path(video).stat().st_size).encode()).hexdigest()
 if cache.exists():
  obj=json.loads(cache.read_text())
  if obj.get('signature')==sig:return obj['tracks']
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);tracks=[];active=[];f=0
 cv2.setNumThreads(2)
 while True:
  ok,im=cap.read()
  if not ok:break
  ds=capsules(im,f/fps);used=set();nxt=[]
  if ds and active:
   costs=np.full((len(active),len(ds)),1e6);swaps={}
   for i,tr in enumerate(active):
    ss=tr['samples'];last=ss[-1];dt=f/fps-last['time'];v=np.array([0.,960.])
    if len(ss)>2:v=np.clip((np.array(last['head'])-ss[-3]['head'])/max(.02,last['time']-ss[-3]['time']),-2200,2200)
    prediction=np.array(last['head'])+v*dt
    for j,q in enumerate(ds):
     direct=np.linalg.norm(q['head']-prediction);reverse=np.linalg.norm(q['tail']-prediction)
     swap=reverse+10<direct;dist=min(direct,reverse+10)
     if dist<70 and abs(q['width']-last['width'])<16:
      costs[i,j]=dist+abs(q['width']-last['width']);swaps[i,j]=swap
   ii,jj=linear_sum_assignment(costs)
   for i,j in zip(ii,jj):
    if costs[i,j]>=1e6:continue
    tr=active[i];q=ds[j]
    if swaps[i,j]:q['head'],q['tail']=q['tail'],q['head']
    tr['samples'].append(serial(q));nxt.append(tr);used.add(j)
  for j,q in enumerate(ds):
   if j not in used:
    tr={'id':600000+len(tracks),'samples':[serial(q)]};tracks.append(tr);nxt.append(tr)
  for tr in active:
   if not any(tr is q for q in nxt) and f/fps-tr['samples'][-1]['time']<.07:nxt.append(tr)
  active=nxt;f+=1
  if f%900==0:print('V6 full capsules',f,flush=True)
 cap.release();tracks=[tr for tr in tracks if len(tr['samples'])>=3];cache.write_text(json.dumps({'signature':sig,'tracks':tracks},separators=(',',':')));return tracks

def serial(q):return {**q,'head':q['head'].tolist(),'tail':q['tail'].tolist()}

def stable_caps(s,cap):
 """Keep cap identity through small/occluded bodies instead of flipping it.

 A pinned head can be hidden by its own particles. Only restore its previous
 position when the original frame still has a continuous blue body there.
 """
 s=copy.deepcopy(s);fixed=0
 for j in range(1,len(s)):
  q=s[j];old=s[j-1];axis=np.array(old['tail'])-old['head'];d=np.array(q['tail'])-q['head']
  if axis@d<0:
   q['head'],q['tail']=q['tail'],q['head'];fixed+=1
  if j<5:continue
  prev=np.array([v['head'] for v in s[j-5:j]]);ref=np.median(prev,axis=0)
  if np.max(np.linalg.norm(prev-ref,axis=1))>12:continue
  direction=ref-np.array(q['tail']);direction/=max(1,np.linalg.norm(direction));shift=np.array(q['head'])-ref
  if shift@direction>-15 or np.linalg.norm(shift-(shift@direction)*direction)>12:continue
  cap.set(cv2.CAP_PROP_POS_MSEC,q['time']*1000);ok,im=cap.read()
  if not ok:continue
  pp=ref+(np.array(q['head'])-ref)*np.linspace(0,.9,18)[:,None]
  xx=pp[:,0].round().astype(int).clip(0,1279);yy=pp[:,1].round().astype(int).clip(0,719);pix=im[yy,xx].astype(float)
  # Violet particles darken the capsule at the pinned cap. They still leave
  # blue coverage along the axis, although its green/red channels fall below
  # the clean-body segmentation threshold.
  blue=(pix[:,0]>220)&(pix[:,1]>135)&(pix[:,2]>125)&(pix[:,0]-pix[:,1]>6)
  if blue.mean()>.65:q['head']=ref.tolist();fixed+=1
 return s,fixed

def refine(video,root,events,recover=True):
 tracks=scan(video,root);changes=0;tilted=0;recovered=0;used=set();report=[];cap=cv2.VideoCapture(str(video));endpoint_repairs=0
 # Assign all fragments against the original paths before changing those
 # paths. The former first-fragment-wins rule discarded later body samples.
 groups={};independent=[]
 for tr in tracks:
  p=np.array([[q['time'],*q['head']] for q in tr['samples']]);options=[]
  for i,e in enumerate(events):
   ep=np.asarray(e.get('bodyPath',e.get('path',[])),float)
   if len(ep)<2:continue
   pp=p[(p[:,0]>=ep[0,0]-.005)&(p[:,0]<=ep[-1,0]+.005)]
   if len(pp)<3:continue
   ds=np.hypot(pp[:,1]-np.interp(pp[:,0],ep[:,0],ep[:,1]),pp[:,2]-np.interp(pp[:,0],ep[:,0],ep[:,2]))
   if (ds<36).sum()>=3 and (ds<36).mean()>.60:options.append((float(np.median(ds))+15*(e['duration']<=0),i))
  if options:groups.setdefault(min(options)[1],[]).append(tr)
  else:independent.append(tr)
 joined=[]
 for i,parts in groups.items():
  samples={}
  for tr in sorted(parts,key=lambda q:len(q['samples']),reverse=True):
   for q in tr['samples']:samples.setdefault(q['time'],q)
  joined.append({'id':parts[0]['id'],'samples':[samples[t] for t in sorted(samples)],'eventIndex':i})
 for tr in joined+independent:
  s,repairs=stable_caps(tr['samples'],cap);endpoint_repairs+=repairs;tr={**tr,'samples':s}
  s=tr['samples'];p=np.array([[q['time'],*q['head']] for q in s]);tails=np.array([[q['time'],*q['tail']] for q in s]);options=[]
  if 'eventIndex' in tr:options=[(0,tr['eventIndex'])]
  for i,e in enumerate(events):
   if 'eventIndex' in tr:break
   ep=np.asarray(e.get('bodyPath',e.get('path',[])),float)
   if len(ep)<2:continue
   pp=p[(p[:,0]>=ep[0,0]-.005)&(p[:,0]<=ep[-1,0]+.005)]
   if len(pp)<3:continue
   ds=np.hypot(pp[:,1]-np.interp(pp[:,0],ep[:,0],ep[:,1]),pp[:,2]-np.interp(pp[:,0],ep[:,0],ep[:,2]))
   close=(ds<36).sum()
   if close>=3 and close/len(pp)>.60:options.append((float(np.median(ds))+15*(e['duration']<=0),i))
  if options:
   _,i=min(options);e=events[i]
   # A track is allowed to replace an old hold only when it extends its
   # currently wrong measurements, not a lone unrelated future capsule.
   if i in used:continue
  else:
   if not recover or len(s)<5:continue
   # Recover only a visible approach to a target with fresh hit evidence;
   # provisional fake candidates are assigned by the normal counter solver.
   end=p[-1,0]+1/30;t=end-.15;hp=p[-1,1:];velocity=np.diff(p[:,1:],axis=0)/np.diff(p[:,0])[:,None];speed=float(np.median(np.linalg.norm(velocity,axis=1)))
   if speed<180 or speed>2400:continue
   e={'time':round(t,5),'sourceTime':round(t,5),'x':float(hp[0]),'sourceX':float(hp[0]),'judgeY':float(hp[1]/720),'type':0,'duration':.15,'isFake':True,'speed':speed,'frames':len(s),'track':tr['id'],'evidence':'independent-full-resolution-capsule','confidence':'low','rotation':90.}
   i=len(events);events.append(e);recovered+=1
  used.add(i);old=(e['time'],e['duration']);t=e['time'];end=t+e['duration']
  # Recover the source-held interval from deceleration of the physical cap.
  vel=np.diff(p[:,1:],axis=0)/np.diff(p[:,0])[:,None];speeds=np.linalg.norm(vel,axis=1)
  axis=p[:-1,1:]-tails[:-1,1:];axis/=np.maximum(1,np.linalg.norm(axis,axis=1))[:,None];normal_speed=np.sum(vel*axis,axis=1)
  contacts=[]
  for j in range(3,len(p)-2):
   # Contact caps can wobble under overlapping particles by several pixels.
   # Require sustained axial deceleration rather than one noiseless frame.
   settled=np.median(normal_speed[j:min(len(normal_speed),j+3)])
   if np.median(normal_speed[max(0,j-3):j])>350 and abs(normal_speed[j])<350 and abs(settled)<250 and abs(p[j,0]-t)<(.8 if e.get('isFake') else .35):contacts.append(j)
  if contacts:
   j=contacts[0];prior=np.median(vel[max(0,j-3):j],axis=0);point=p[j,1:].copy();dt=float((point-p[j-1,1:])@prior/max(1,prior@prior));t=float(np.clip(p[j-1,0]+dt,p[j-1,0],p[j,0]+.01))
   e['sourceContactTime']=round(t,5);e['capsuleContactTime']=round(t,5);e['capsuleContactPoint']=point.tolist();e['x']=float(point[0]);e['judgeY']=float(point[1]/720)
  # Detect tail collapse from actual cap separation. The final visible
  # capsule often ends BEFORE its guessed old end; infer only a few frames.
  lengths=np.linalg.norm(tails[:,1:]-p[:,1:],axis=1);last=s[-1]
  if p[-1,0]>t+.015 and not last['clipped']:
   near=np.flatnonzero((p[:,0]>=max(t,p[-1,0]-.17))&np.array([not q['clipped'] for q in s]))
   if len(near)>=3:
    v=-float(np.median(np.diff(lengths[near])/np.diff(p[near,0])))
    if 150<v<3000:
     finish=p[-1,0]+min(.15,lengths[-1]/v)
     if abs(finish-end)<.65 or contacts:end=float(finish)
   if contacts and end<p[-1,0]:end=float(p[-1,0]+1/30)
  if end<=t+.02:end=t+.03
  e['type']=0;e['time']=round(t,5);e['duration']=round(end-t,5);e['fitBaseTime']=e['time'];e['fitBaseDuration']=e['duration']
  # Extrapolate a clipped upper tail from the first uncut source measurements
  # instead of treating the camera/HUD boundary as a real capsule cap.
  valid=np.flatnonzero([not q['clipped'] for q in s])
  for j,q in enumerate(s):
   if q['clipped'] and q['tail'][1]<40 and len(valid)>=3:
    near=valid[valid>j][:6]
    if len(near)>=3:
     vv=np.median(np.diff(tails[near,1:],axis=0)/np.diff(tails[near,0])[:,None],axis=0);tails[j,1:]=tails[near[0],1:]+vv*(p[j,0]-p[near[0],0]);d=np.array(q['tail'])-q['head'];d/=max(1,np.linalg.norm(d));tails[j,1:]=p[j,1:]+d*max(1,float((tails[j,1:]-p[j,1:])@d))
  e['path']=p.tolist();e['bodyPath']=p.tolist();e['tailPath']=tails.tolist();e['frames']=len(s);e['bodyEvidence']='full-resolution-blue-capsule-cap-pair';e['holdBodyWidth']=float(np.median([q['width'] for q in s]))
  e['holdWidthPath']=[[q['time'],q['width']] for q in s]
  angles=capsule_rotations(p,tails)
  e['holdRotationPath']=[[float(tm),float(a)] for tm,a in zip(p[:,0],angles)];e['rotation']=float(np.interp(t,p[:,0],angles))
  if abs(e['rotation']-90)>2 or np.ptp(angles)>3:tilted+=1
  if e['duration']>0 and not e['isFake']:
   after=p[(p[:,0]>t)&(p[:,0]<=end)]
   # Moving physical cap plus a nearby circle/ring supplies the emitter path.
   # Final source ring fitting refines it, then merge the physical samples.
   e['capsuleAnchorPath']=[[t,e['x'],e['judgeY']*720]]+after.tolist()+[[end,*p[-1,1:].tolist()]]
  report.append({'track':tr['id'],'event':e.get('track'),'previous':list(old),'fitted':[e['time'],e['duration']],'samples':len(s),'rotationRange':[float(min(angles)),float(max(angles))]});changes+=1
 cap.release()
 result={'measuredTracks':len(tracks),'capsulesRefitted':changes,'tiltedOrRotatingHolds':tilted,'independentCandidatesRecovered':recovered,'endpointIdentityOrOcclusionRepairs':endpoint_repairs,'joinedCapsuleFragments':sum(len(p)-1 for p in groups.values()),'changes':report}
 (Path(root)/'hold-motion-v6.json').write_text(json.dumps(result,indent=2));print('V6 capsules', {k:v for k,v in result.items() if k!='changes'},flush=True);return result

def apply_anchors(events):
 for e in events:
  pp=e.get('capsuleAnchorPath')
  if not pp or e['isFake'] or e['duration']<=0:continue
  t=e['time'];end=t+e['duration'];p=np.array(pp,float);old=np.array(e.get('anchorPath',[[t,e['x'],e['judgeY']*720]]),float);out=[[t,e['x'],e['judgeY']*720]]
  for tm,x,y in p:
   if not t+.01<tm<end:continue
   # Prefer matching effect centres; use the measured cap when the ring is
   # occluded. Detector cap jitter cannot teleport a judgement emitter.
   ax=float(np.interp(tm,old[:,0],old[:,1]));ay=float(np.interp(tm,old[:,0],old[:,2]))
   if math.hypot(ax-x,ay-y)<12:x,y=ax,ay
   if math.hypot(x-out[-1][1],y-out[-1][2])>max(30,1600*(tm-out[-1][0])):continue
   out.append([float(tm),float(x),float(y)])
  out.append([end,out[-1][1],out[-1][2]]);e['anchorPath']=out

def merge_repairs(events):
 """A measured hold head/tail replaces an earlier effect-only tap guess."""
 contacts=[]
 for e in events:
  if not e.get('capsuleContactTime') or not e.get('ringContactSupport'):continue
  p=np.asarray(e.get('bodyPath',[]),float)
  if len(p)<3:continue
  for tm in (e['time'],e['time']+e['duration']):
   contacts.append((tm,*[float(np.interp(tm,p[:,0],p[:,k])) for k in (1,2)],e))
 removed=[]
 for e in events:
  if e.get('duration') or e.get('evidence')!='video-counter-and-fresh-source-hit-ring' or e.get('frames',0)>3:continue
  matches=[q for q in contacts if abs(e['time']-q[0])<.105 and math.hypot(e['x']-q[1],e['judgeY']*720-q[2])<38]
  if matches:removed.append(e)
 for e in removed:events.remove(e)
 return {'effectOnlyTapGuessesMergedIntoMeasuredHold':len(removed)}
