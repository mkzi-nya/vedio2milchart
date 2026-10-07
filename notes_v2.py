"""Full-screen note tracking with dynamic circle intersections, video pixels only."""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

def scan(video,root,faint=False):
 root=Path(root);target_tracks=json.loads((root/'targets-v2.json').read_text());targets={}
 for tr in target_tracks:
  if tr['samples'][0]['radius']>28:continue
  for q in tr['samples']:targets.setdefault(round(q['time']*30),[]).append((q['x']/2,q['y']/2,tr['id'],q['alpha']))
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30;f=0;tracks=[];active=[]
 while True:
  ok,raw=cap.read()
  if not ok:break
  im=cv2.resize(raw,(640,360),interpolation=cv2.INTER_AREA);t=f/fps
  # Hit petals and rings must never join a bright capsule into a second note.
  # Faint decorative outlines are supplied by the independent faint pass.
  mask=(((im[:,:,0]>170)&(im[:,:,1]>145)&(im[:,:,2]>155)) if faint else ((im[:,:,0]>215)&(im[:,:,1]>175)&(im[:,:,2]>175))).astype(np.uint8);mask[:51]=0
  mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
  n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8);ds=[]
  for k in range(1,n):
   x0,y0,w,h,area=stats[k]
   if area<32 or min(w,h)<8 or min(w,h)>31 or max(w,h)>355:continue
   piece=(labels[y0:y0+h,x0:x0+w]==k);fill=area/(w*h);pix=im[y0:y0+h,x0:x0+w][piece].astype(float);chroma=float(np.mean(pix.max(1)-pix.min(1)))
   held=max(w,h)>34
   if not held and not .43<w/h<2.1:continue
   if not held and .9<w/h<1.12 and fill<.59 and chroma<22:continue
   if held and fill<.42:continue
   cx,cy=(x0+(w-1)/2,y0+(h-1)/2);tx,ty=cx,cy
   if held:
    yy,xx=np.nonzero(piece);rect=cv2.minAreaRect(np.column_stack([xx+x0,yy+y0]).astype(np.float32));corners=cv2.boxPoints(rect);lens=[np.linalg.norm(corners[(j+1)%4]-corners[j]) for j in range(4)];j=int(np.argmin(lens));a=(corners[j]+corners[(j+1)%4])/2;b=(corners[(j+2)%4]+corners[(j+3)%4])/2;short=min(lens);axis=(b-a)/max(.01,np.linalg.norm(b-a));a=a+axis*short/2;b=b-axis*short/2
    near=targets.get(round(t*30),[])
    if abs(a[0]-b[0])<8 and abs(a[1]-b[1])>15:
     head,tail=(a,b) if a[1]>b[1] else (b,a)
    elif near:
     da=min(math.hypot(a[0]-q[0],a[1]-q[1]) for q in near);db=min(math.hypot(b[0]-q[0],b[1]-q[1]) for q in near);head,tail=(a,b) if da<db else (b,a)
    else:head,tail=(a,b) if a[1]>b[1] else (b,a)
    cx,cy=map(float,head);tx,ty=map(float,tail)
   typ=0 if held or fill>.62 else 1
   ds.append([round(t,5),float(cx),float(cy),int(w),int(h),typ,int(held),float(fill),int(y0),float(tx),float(ty),chroma])
  used=set();nxt=[]
  if ds and active:
   costs=np.ones((len(active),len(ds)))*1e6
   for i,tr in enumerate(active):
    s=tr['samples'];last=s[-1];dt=t-last[0];vx,vy=0,480
    if len(s)>=3:
     prev=s[-3];dd=last[0]-prev[0]
     if dd>0:vx=np.clip((last[1]-prev[1])/dd,-1200,1200);vy=np.clip((last[2]-prev[2])/dd,-1200,1200)
    if last[6] and len(s)>3 and abs(vy)<35:vy=0
    for j,d in enumerate(ds):
     dx=d[1]-last[1]-vx*dt;dy=d[2]-last[2]-vy*dt;dist=math.hypot(dx,dy)
     if dist<30+dt*50:costs[i,j]=dist+8*(d[5]!=last[5])+6*(d[6]!=last[6])
   ii,jj=linear_sum_assignment(costs)
   for i,j in zip(ii,jj):
    if costs[i,j]>=1e6:continue
    tr=active[i];tr['samples'].append(ds[j]);nxt.append(tr);used.add(j)
  for j,d in enumerate(ds):
   if j not in used:tr={'id':len(tracks),'samples':[d]};tracks.append(tr);nxt.append(tr)
  for tr in active:
   if not any(q is tr for q in nxt) and t-tr['samples'][-1][0]<.095:nxt.append(tr)
  active=nxt;f+=1
  if f%900==0:print('notes-v2:',f,'frames',flush=True)
 cap.release();events=[]
 for tr in tracks:
  s=np.asarray(tr['samples']);
  if len(s)<4:continue
  holds=s[:,6].sum()>=3
  # Estimate moving-head speed; stationary held heads are excluded from the fit.
  delta=np.diff(s[:,:3],axis=0);vel=np.linalg.norm(delta[:,1:3],axis=1)/np.maximum(delta[:,0],.001);moving=np.flatnonzero((vel>70)&(vel<1500))
  if not len(moving):continue
  speed=float(np.median(vel[moving])*2)
  if not 130<speed<2600:continue
  candidates=[]
  for k in range(3,len(s)):
   window=s[max(0,k-4):k+1];dt=window[-1,0]-window[0,0]
   if dt<=0:continue
   vx,vy=(window[-1,1:3]-window[0,1:3])/dt;v2=vx*vx+vy*vy
   if v2<70**2:continue
   last=s[k]
   nearby=targets.get(round(last[0]*30),[])
   for x,y,ident,alpha in nearby:
    if alpha<.2:continue
    jump=((x-last[1])*vx+(y-last[2])*vy)/v2
    residual=math.hypot(last[1]+vx*jump-x,last[2]+vy*jump-y)
    if not -.07<jump<.14 or residual>14 or abs(y-last[2])>65:continue
    hit=last[0]+jump
    future=targets.get(round(hit*30),[])
    if future:
     matches=[q for q in future if q[2]==ident]
     if matches:
      x,y,_,alpha=matches[0];jump=((x-last[1])*vx+(y-last[2])*vy)/v2;hit=last[0]+jump;residual=math.hypot(last[1]+vx*jump-x,last[2]+vy*jump-y)
    if residual>18:continue
    candidates.append((residual+abs(jump)*8+max(0,jump)*2,hit,x,y,ident,k,vx,vy))
  if candidates:
   best=min(candidates,key=lambda q:q[0]);_,hit,x,y,ident,k,vx,vy=best;fake=False
   if holds:
    # A held head stays on its moving target; use the first approach intersection.
    close=[q for q in candidates if q[0]<12 and q[7]>100] or [best]
    _,hit,x,y,ident,k,vx,vy=min(close,key=lambda q:q[1])
  else:
   hit=float(s[-1,0]+1/30);x,y=map(float,s[-1,1:3]);ident=None;k=len(s)-1;fake=True;window=s[-min(6,len(s)):];vx,vy=(window[-1,1:3]-window[0,1:3])/max(.01,window[-1,0]-window[0,0])
  if not 0<=hit<f/fps:continue
  duration=0
  if holds:
   body=s[s[:,6]>0];length=np.linalg.norm(body[:,9:11]-body[:,1:3],axis=1)*2
   uncut=(body[:,8]>1)&(body[:,8]+body[:,4]<359)
   physical=float(np.percentile(length[uncut],65)/speed) if uncut.any() else float(np.percentile(length,75)/speed)
   if not fake:
    pinned=body[(body[:,0]>hit)&(np.linalg.norm(body[:,1:3]-[x,y],axis=1)<25)]
    duration=physical
    length_all=np.linalg.norm(s[:,9:11]-s[:,1:3],axis=1)*2
    after=np.flatnonzero((s[:,0]>hit)&(length_all>14))
    if len(after):
     j=after[-1];finish=float(s[j,0]+length_all[j]/max(300,speed))
     if j+1<len(s) and length_all[j+1]<length_all[j]:finish=float(s[j,0]+length_all[j]/max(1,length_all[j]-length_all[j+1])*(s[j+1,0]-s[j,0]))
     duration=max(.03,finish-hit)
   else:duration=physical
  if not holds and np.ptp(s[:,1:3],axis=0).max()*2<65:continue
  typ=0 if holds else int(np.median(s[:,5])>=.5)
  events.append({'time':round(float(hit),5),'sourceTime':round(float(hit),5),'x':round(float(x*2),2),'sourceX':round(float(x*2),2),'judgeY':round(float(y/360),6),'type':typ,'duration':round(duration,5),'frames':len(s),'track':tr['id'],'speed':round(speed,3),'isFake':fake,'confidence':'medium' if len(s)>=8 and candidates else 'low','evidence':'dynamic-target-intersection' if candidates else 'full-screen-decorative-track','targetTrack':ident,'rotation':90. if holds and np.median(abs(s[:,9]-s[:,1]))<6 else round(float((90+math.degrees(math.atan2(vx,vy)))%360),4),'path':[[float(q[0]),float(q[1]*2),float(q[2]*2)] for q in s[::2]],'tailPath':[[float(q[0]),float(q[9]*2),float(q[10]*2)] for q in s[::2]] if holds else []})
 events.sort(key=lambda e:e['time']);clean=[]
 for e in sorted(events,key=lambda e:e['frames'],reverse=True):
  if any(abs(e['time']-q['time'])<.06 and abs(e['x']-q['x'])<34 and abs(e['judgeY']-q['judgeY'])<.04 for q in clean):continue
  clean.append(e)
 clean.sort(key=lambda e:e['time']);(root/('notes-faint-v5.json' if faint else 'notes-v2.json')).write_text(json.dumps(clean,ensure_ascii=False,indent=2))
 if not faint:(root/'tracks-v2.json').write_text(json.dumps(tracks,separators=(',',':')))
 print({'tracks':len(tracks),'notes':len(clean),'targetMatched':sum(not e['isFake'] for e in clean),'holds':sum(e['duration']>0 for e in clean),'faintPass':faint},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',required=True);a=p.parse_args();scan(a.video,a.output)
