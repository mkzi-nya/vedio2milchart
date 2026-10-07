"""Fuse video-derived geometry, tracks and counter constraints for playable export."""
import json,math,hashlib
from pathlib import Path
import numpy as np,cv2
from scipy import ndimage
from timeline import refine,settle

def clean_geometry(root,video=None):
 root=Path(root);targets=json.loads((root/'targets-v2.json').read_text());rails=json.loads((root/('rails-v3.json' if (root/'rails-v3.json').exists() else 'rails-v2.json')).read_text())
 if (root/'ghosts-v3.json').exists():targets=[tr for tr in targets if tr['samples'][0]['radius']<28]+json.loads((root/'ghosts-v3.json').read_text())
 if video is not None:
  from geometry_fill import fill
  fill_report=fill(video,root,targets,rails,scan_ghosts=not (root/'ghosts-v3.json').exists());(root/'geometry-fill-report.json').write_text(json.dumps(fill_report))
 for tr in targets:
  s=tr['samples']
  if len(s)>=7:
   smooth=ndimage.median_filter(np.array([[q['x'],q['y']] for q in s]),size=(5,1),mode='nearest')
   for q,(x,y) in zip(s,smooth):q['x']=float(x);q['y']=float(y)
 frames={}
 for tr in targets:
  if tr['samples'][0]['radius']>28:continue
  for q in tr['samples']:frames.setdefault(round(q['time']*30),[]).append(q)
 edges={0:[],1280:[]}
 for frame,ss in frames.items():
  for row in {round(q['y']/8)*8 for q in ss}:
   qs=sorted([q for q in ss if abs(q['y']-row)<5],key=lambda q:q['x'])
   if len(qs)<5:continue
   gaps=np.diff([q['x'] for q in qs]);step=float(np.median(gaps))
   if not 140<step<180 or np.median(abs(gaps-step))>4:continue
   y=float(np.median([q['y'] for q in qs]));alpha=float(np.median([q['alpha'] for q in qs]));r=qs[0]['radius']
   for x,valid in [(0,abs(qs[0]['x']-step)<8),(1280,abs(1280-qs[-1]['x']-step)<8)]:
    if valid:edges[x].append({'time':frame/30,'x':float(x),'y':y,'radius':r,'alpha':alpha})
 for x,ss in edges.items():
  ss.sort(key=lambda q:q['time']);parts=[]
  for q in ss:
   if not parts or q['time']-parts[-1][-1]['time']>.18:parts.append([])
   parts[-1].append(q)
  for s in parts:
   if len(s)>=3:targets.append({'id':100000+len(targets),'kind':'target','samples':s,'evidence':'partial-edge-head-grid'})
 # A Hough segment can include the far arc of its endpoint circle. Stop at the near rim.
 for tr in rails:
  for q in tr['samples']:
   ends=np.array(q.get('ends',[]),float)
   if ends.size!=4:continue
   ends=ends.reshape(2,2)
   for j in (0,1):
    delta=ends[j]-ends[1-j];length=np.linalg.norm(delta)
    if length<30:continue
    axis=delta/length;near=frames.get(round(q['time']*30),[]);matches=[]
    for head in near:
     c=np.array([head['x'],head['y']]);dist=np.linalg.norm(c-ends[j]);perp=abs(np.cross(axis,c-ends[j]))
     if head['alpha']>.2 and dist<head['radius']+18 and perp<9:matches.append((dist,head,c))
    if matches:
     _,head,c=min(matches,key=lambda v:v[0]);ends[j]=c-axis*head['radius']
   delta=ends[1]-ends[0];q['x'],q['y']=map(float,ends.mean(0));q['rotation']=math.degrees(math.atan2(delta[1],delta[0]));q['width']=float(np.linalg.norm(delta)*1.5/512)
 if video is not None:
  from line_reconstruct_v6 import scan as scan_lines
  effects,motion_report=scan_lines(video,root,targets)
  from streak_reconstruct_v1 import scan as scan_streaks
  streaks,streak_report=scan_streaks(video,root)
 else:
  effects=json.loads((root/'lines-v6.json').read_text())['tracks'] if (root/'lines-v6.json').exists() else []
  motion_report={'cachedOnly':True}
  cached_streaks=root/'streaks-v1.json'
  streaks=json.loads(cached_streaks.read_text()).get('tracks',[]) if cached_streaks.exists() else []
  streak_report={'cachedOnly':True,'tracks':len(streaks)}
 effects.extend(streaks)
 (root/'line-motion-v5.json').write_text(json.dumps(motion_report,indent=2))
 (root/'streak-review-v1.json').write_text(json.dumps(streak_report,indent=2))
 effects.extend({'kind':'target','samples':tr['samples'],'id':tr['id']} for tr in targets)
 (root/'raw-target-effects-v6.json').write_text(json.dumps([q for q in effects if q['kind']=='target'],separators=(',',':')))
 if video is not None:
  from target_review_v6 import refine as refine_targets
  effects,target_review=refine_targets(video,root,effects)
  (root/'native-target-review-v6.json').write_text(json.dumps(target_review,indent=2))
 (root/'visuals-v2.json').write_text(json.dumps(effects,separators=(',',':')))
 return targets,effects

def enrich(video,root):
 root=Path(root);targets,effects=clean_geometry(root,video);es=json.loads((root/'notes-v2.json').read_text());old=json.loads((root/('events-v1.json' if (root/'events-v1.json').exists() else 'events.json')).read_text());supplements=0
 faint_supplements=0
 if (root/'notes-faint-v5.json').exists():
  for e in json.loads((root/'notes-faint-v5.json').read_text()):
   ep=np.asarray(e['path'],float)
   if e['frames']<6:continue
   shared=False
   for other in es:
    op=np.asarray(other.get('path',[]),float)
    if len(op)<2:continue
    common=ep[(ep[:,0]>=op[0,0])&(ep[:,0]<=op[-1,0])]
    if len(common)<2:continue
    ds=np.hypot(common[:,1]-np.interp(common[:,0],op[:,0],op[:,1]),common[:,2]-np.interp(common[:,0],op[:,0],op[:,2]))
    if (ds<24).sum()>=2:shared=True;break
   if shared:continue
   e['evidence']='independent-faint-source-approach';es.append(e);faint_supplements+=1
 for e in old:
  e={**e,'time':max(0,e['time']-1/30),'sourceTime':max(0,e.get('sourceTime',e['time'])-1/30),'path':[[max(0,q[0]-1/30),q[1],q[2]] for q in e.get('path',[])]}
  if e['isFake'] or e.get('frames',0)<5:continue
  ep=np.asarray(e.get('path',[]),float)
  if len(ep)<2 or np.ptp(ep[:,1:3],axis=0).max()<60:continue
  shared=False
  for other in es:
   op=np.asarray(other.get('path',[]),float)
   if len(op)<2 or op[-1,0]<ep[0,0] or op[0,0]>ep[-1,0]:continue
   common=ep[(ep[:,0]>=op[0,0])&(ep[:,0]<=op[-1,0])]
   if len(common)<2:continue
   distances=np.hypot(common[:,1]-np.interp(common[:,0],op[:,0],op[:,1]),common[:,2]-np.interp(common[:,0],op[:,0],op[:,2]))
   if (distances<22).sum()>=2:shared=True;break
  if shared:continue
  if any(abs(e['time']-q['time'])<.11 and abs(e['x']-q['x'])<45 for q in es):continue
  es.append({**e,'evidence':'occluded-independent-pass','confidence':'low'});supplements+=1
 cap=cv2.VideoCapture(str(video));samples={}
 for i,e in enumerate(es):
  for t,x,y in e.get('path',[]):samples.setdefault(round(t*30),[]).append((i,t,x,y))
 maxframe=max(samples,default=0);frame=0;directions={};widths={};auras={}
 while frame<=maxframe:
  ok,im=cap.read()
  if not ok:break
  for i,t,x,y in samples.get(frame,[]):
   e=es[i];xi,yi=round(x),round(y);radius=32;x0=max(0,xi-radius);y0=max(0,yi-radius);crop=im[y0:min(720,yi+radius),x0:min(1280,xi+radius)]
   if crop.size==0:continue
   mask=((crop[:,:,0]>170)&(crop[:,:,1]>140)&(crop[:,:,2]>150)).astype(np.uint8)*255;contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   candidates=[c for c in contours if cv2.contourArea(c)>100 and cv2.pointPolygonTest(c,(float(xi-x0),float(yi-y0)),False)>=0]
   if not candidates:continue
   c=max(candidates,key=cv2.contourArea);xx,yy,w,h=cv2.boundingRect(c)
   if e['duration']<=0 and max(w,h)<62:widths.setdefault(i,[]).append((t,(w+h)/2))
   if e['type']==0 and e['duration']<=0 and t<e['time']-.15 and y>140 and y<e['judgeY']*720-75 and 48<x<1232:
    phi=np.arange(32)*math.pi/16;inner=[];outer=[]
    for a0 in phi:
     iy=int(round(y+25*math.sin(a0)));ix=int(round(x+25*math.cos(a0)));oy=int(round(y+39*math.sin(a0)));ox=int(round(x+39*math.cos(a0)))
     if 0<=iy<720 and 0<=oy<720:inner.append(float(im[iy,ix].min()));outer.append(float(im[oy,ox].min()))
    if inner:auras.setdefault(i,[]).append(float(np.percentile(inner,30)-np.median(outer)))
   if e['type']!=1 or e['duration']>0:continue
   approx=cv2.approxPolyDP(cv2.convexHull(c),2.2,True).reshape(-1,2).astype(float)
   if len(approx)<4:continue
   angles=[]
   for k,p in enumerate(approx):
    a=approx[k-1]-p;b=approx[(k+1)%len(approx)]-p;angles.append(math.acos(float(np.clip(np.dot(a,b)/max(.001,np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
   tip=approx[int(np.argmin(angles))];center=approx.mean(0);angle=math.degrees(math.atan2(tip[1]-center[1],tip[0]-center[0]));directions.setdefault(i,[]).append([t,angle])
  frame+=1
 cap.release()
 for i,e in enumerate(es):
  if len(auras.get(i,[]))>=3:
   e['auraEvidence']=round(float(np.median(auras[i])),3)
  if i in directions:
   q=directions[i];angles=np.unwrap(np.radians([v[1] for v in q]));e['orientationPath']=[[v[0],round(float(a*180/math.pi),3)] for v,a in zip(q,angles)]
  if i in widths:e['noteScale']=float(np.clip(np.median([q[1] for q in widths[i]])/(2000*.0223*(335/185)*.54),.6,1.3))
 # Keep source observations intact for the dense per-frame/effect solver.
 intervals=[];allocation={'state':'awaiting-source-ring-and-frame-counter-fit'}
 es.sort(key=lambda e:e['time']);(root/'events-v2.json').write_text(json.dumps(es,ensure_ascii=False,indent=2));(root/'counter-v2.json').write_text(json.dumps({'intervals':intervals,'allocation':allocation},ensure_ascii=False,indent=2))
 cap=cv2.VideoCapture(str(video));frames=[]
 for t in np.linspace(3,137,55):
  cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,im=cap.read()
  if ok:frames.append(im)
 cap.release();bg=np.percentile(np.stack(frames),3,axis=0).astype(np.uint8);mask=np.zeros(bg.shape[:2],np.uint8);mask[:112]=255;bg=cv2.inpaint(bg,mask,5,cv2.INPAINT_TELEA)
 (root/'storyboard').mkdir(exist_ok=True);cv2.imwrite(str(root/'storyboard/background.jpg'),bg);cv2.imwrite(str(root/'cover.jpg'),bg)
 return es,effects,{'candidates':len(es),'judgements':sum(1+(e['duration']>0) for e in es if not e['isFake']),'fakeNotes':sum(e['isFake'] for e in es),'visibleTargets':len(targets),'rails':sum(q.get('kind')=='rail' for q in effects),'streaks':sum(q.get('kind')=='streak' for q in effects),'independentPassSupplements':supplements,'faintNoteSupplements':faint_supplements,'allocation':allocation}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',required=True);a=p.parse_args();_,_,report=enrich(a.video,a.output);print(report,flush=True)
