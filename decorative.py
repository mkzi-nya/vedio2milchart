"""Recover faint hollow moving teardrops, excluded from judgement count fitting."""
import json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

def scan(video,root):
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30;frame=0;tracks=[];active=[]
 while True:
  ok,raw=cap.read()
  if not ok:break
  im=cv2.resize(raw,(640,360));rgb=im.astype(float);lo=rgb.min(2);ch=rgb.max(2)-lo
  mask=((lo>65)&(rgb[:,:,0]>85)&(rgb[:,:,0]>=rgb[:,:,1]+6)&(ch>9)&(ch<115)).astype(np.uint8)*255;mask[:50]=0
  contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);ds=[];t=frame/fps
  if hierarchy is not None:
   for i,c in enumerate(contours):
    if hierarchy[0,i,3]>=0 or hierarchy[0,i,2]<0:continue
    x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(c);hole=cv2.contourArea(contours[hierarchy[0,i,2]])
    if not(8<=min(w,h)<=26 and max(w,h)<=30 and .45<w/h<2.1 and 30<area<500 and .12<hole/max(area,1)<.85):continue
    hull=cv2.approxPolyDP(cv2.convexHull(c),1,True).reshape(-1,2).astype(float)
    angles=[]
    for j,p in enumerate(hull):
     a=hull[j-1]-p;b=hull[(j+1)%len(hull)]-p;angles.append(math.acos(float(np.clip(np.dot(a,b)/max(.001,np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
    tip=hull[int(np.argmin(angles))];center=np.array([x+(w-1)/2,y+(h-1)/2]);angle=math.degrees(math.atan2(*(tip-center)[::-1]));roi=lo[y:y+h,x:x+w];ring=mask[y:y+h,x:x+w]>0;bg=float(np.percentile(roi,20));alpha=float(np.clip((np.percentile(roi[ring],70)-bg)/max(1,210-bg),.08,1));ds.append([t,*center,w,h,angle,alpha])
  used=set();nxt=[]
  if ds and active:
   costs=np.ones((len(active),len(ds)))*1e6
   for i,tr in enumerate(active):
    s=np.asarray(tr['samples']);last=s[-1];v=np.zeros(2)
    if len(s)>2:v=np.clip((last[1:3]-s[-3,1:3])/max(.01,last[0]-s[-3,0]),-900,900)
    for j,d in enumerate(ds):
     dist=np.linalg.norm(np.asarray(d[1:3])-last[1:3]-v*(t-last[0]))
     if dist<22:costs[i,j]=dist
   ii,jj=linear_sum_assignment(costs)
   for i,j in zip(ii,jj):
    if costs[i,j]>1000:continue
    tr=active[i];tr['samples'].append(ds[j]);nxt.append(tr);used.add(j)
  for j,d in enumerate(ds):
   if j not in used:tr={'samples':[d]};tracks.append(tr);nxt.append(tr)
  for tr in active:
   if not any(q is tr for q in nxt) and t-tr['samples'][-1][0]<.11:nxt.append(tr)
  active=nxt;frame+=1
 cap.release();es=[]
 for tr in tracks:
  s=np.asarray(tr['samples']);extent=np.ptp(s[:,1:3],axis=0).max()*2
  if len(s)<8 or extent<45:continue
  vel=np.linalg.norm(np.diff(s[:,1:3],axis=0),axis=1)/np.maximum(.001,np.diff(s[:,0]))*2;speed=float(np.median(vel))
  if not 20<speed<1800:continue
  x,y=s[-1,1:3]*2;t=s[-1,0]+1/30
  es.append({'time':float(t),'sourceTime':float(t),'x':float(x),'sourceX':float(x),'judgeY':float(y/720),'type':1,'duration':0,'frames':len(s),'speed':speed,'isFake':True,'confidence':'inferred','evidence':'faint-hollow-moving-decoration','rotation':90,'path':[[float(q[0]),float(q[1]*2),float(q[2]*2)] for q in s[::2]],'orientationPath':[[float(q[0]),float(a)] for q,a in zip(s[::2],np.degrees(np.unwrap(np.radians(s[::2,5]))))],'opacityPath':[[float(q[0]),float(q[6])] for q in s[::2]],'noteScale':float(np.clip(np.median((s[:,3]+s[:,4]))/(2000*.0223*(335/185)*.54),.6,1.3))})
 (Path(root)/'decorative-v2.json').write_text(json.dumps(es,ensure_ascii=False));print('faint decorations:',len(es),flush=True)
 return es

def merge(events,decorations):
 kept=[]
 for e in decorations:
  ep=np.asarray(e['path']);shared=False
  for q in events+kept:
   op=np.asarray(q.get('path',[]))
   if len(op)<2 or op[-1,0]<ep[0,0] or op[0,0]>ep[-1,0]:continue
   common=ep[(ep[:,0]>=op[0,0])&(ep[:,0]<=op[-1,0])]
   if len(common)<2:continue
   ds=np.hypot(common[:,1]-np.interp(common[:,0],op[:,0],op[:,1]),common[:,2]-np.interp(common[:,0],op[:,0],op[:,2]))
   if (ds<23).sum()>=2:
    shared=True
    # A faint independent pass often tracks the same waterdrop beyond the
    # bright detector's short fragment. Keep those additional observed frames.
    if q.get('isFake') and q.get('duration',0)<=0 and q.get('type')==1 and (ds<23).mean()>.6:
     extended=[p.tolist() for p in ep if p[0]<op[0,0]-.001]+op.tolist()+[p.tolist() for p in ep if p[0]>op[-1,0]+.001]
     q['path']=extended;q['opacityPath']=e.get('opacityPath',[])
     q['orientationPath']=e.get('orientationPath',q.get('orientationPath',[]))
     q['visualEvidence']='bright-and-faint-independent-track-union'
    break
  if not shared:kept.append(e)
 events.extend(kept);return len(kept)
