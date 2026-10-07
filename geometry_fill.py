"""Recover circle arcs near HUD and extend rail gaps only with supporting video pixels."""
import json,math
from pathlib import Path
import cv2,numpy as np
from geometry import circle_kernel

def fill(video,root,targets,rails,scan_ghosts=True):
 byframe={};rframes={}
 for tr in targets:
  for q in tr['samples']:byframe.setdefault(round(q['time']*30),[]).append(q)
 for tr in rails:
  for q in tr['samples']:rframes.setdefault(round(q['time']*30),[]).append(q)
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30;f=0;tracks=[];active=[];kernel=circle_kernel(19.5);extensions=0
 while True:
  ok,raw=cap.read()
  if not ok:break
  im=cv2.resize(raw,(640,360)).astype(float);lo=im.min(2);ch=im.max(2)-lo;white=(lo*np.clip(1-(ch-10)/40,0,1)).astype(np.float32)
  ds=[];t=f/fps;ys,xs=[],[]
  if scan_ghosts:
   score=cv2.filter2D(white[:112],-1,kernel);mx=cv2.dilate(score,np.ones((9,9),np.uint8));ys,xs=np.nonzero((score>=mx-.001)&(score>8))
  for y,x in zip(ys,xs):
   if not 43<y<87 or x<30 or x>610:continue
   if any(q['radius']>28 and math.hypot(q['x']/2-x,q['y']/2-y)<14 for q in byframe.get(f,[])):continue
   vals=[]
   for a in np.arange(32)*math.pi/16:
    sy=round(y+19.5*math.sin(a));sx=round(x+19.5*math.cos(a));vals.append(white[sy-1:sy+2,sx-1:sx+2].max())
   bg=np.median(lo[y-4:y+5,x-4:x+5]);contrast=np.array(vals)-bg
   if (contrast>12).mean()<.75:continue
   ds.append({'time':t,'x':float(x*2),'y':float(y*2),'radius':39.,'alpha':float(np.clip(np.median(contrast)/220,0,1))})
  nxt=[];used=set()
  for tr in active:
   last=tr['samples'][-1];matches=[(math.hypot(q['x']-last['x'],q['y']-last['y']),j,q) for j,q in enumerate(ds) if j not in used]
   if matches and min(matches)[0]<24:
    _,j,q=min(matches);tr['samples'].append(q);used.add(j);nxt.append(tr)
   elif t-last['time']<.1:nxt.append(tr)
  for j,q in enumerate(ds):
   if j not in used:tr={'kind':'target','id':200000+len(tracks),'samples':[q]};tracks.append(tr);nxt.append(tr)
  active=nxt
  for q in rframes.get(f,[]):
   ends=np.array(q['ends']).reshape(2,2);j=int(np.argmin(ends[:,1]));a=ends[j];b=ends[1-j];delta=b-a
   if a[1]<100 or b[1]-a[1]<100 or abs(delta[0]/max(1,delta[1]))>.035:continue
   # Sample a narrow ridge along the missing continuation and compare local background.
   evidence=[]
   for y in np.linspace(56,min(350,a[1]/2-5),30):
    x=(a[0]+(2*y-a[1])*delta[0]/delta[1])/2;xi=int(round(x));yi=int(round(y))
    if not 5<xi<635 or yi<5 or yi>355:continue
    block=lo[yi-1:yi+2,xi-2:xi+3];excess=block.max()-np.median(lo[yi-3:yi+4,xi-5:xi+6]);evidence.append(excess>12 and ch[yi-1:yi+2,xi-2:xi+3].min()<28)
   if len(evidence)>=10 and np.mean(evidence)>.6:
    ends[j]=[a[0]-a[1]*delta[0]/delta[1],0];q['ends']=ends.flatten().tolist();extensions+=1
  f+=1
 cap.release();extra=[tr for tr in tracks if len(tr['samples'])>=8];targets.extend(extra)
 return {'hudAdjacentCircleTracks':len(extra),'videoSupportedRailExtensions':extensions}
