"""Measure every visible neutral judgement axis directly in every video frame.

Long fast rotations must not depend on the lifetime of a previous Hough track.
Tracks only compact the output; confidence comes from the source ridge pixels.
"""
import json,math,hashlib
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

def detect(im,heads):
 lo=im.min(2).astype(np.float32);ch=im.max(2).astype(np.float32)-lo
 bg=cv2.morphologyEx(lo,cv2.MORPH_OPEN,np.ones((21,21),np.uint8))
 contrast=lo-bg;neutral=ch<29
 mask=((lo>37)&neutral&(contrast>9)).astype('uint8')*255;mask[:112]=0;mask[710:]=0
 lines=cv2.HoughLinesP(mask,1,np.pi/1440,threshold=55,minLineLength=125,maxLineGap=38)
 proposals=[]
 for x1,y1,x2,y2 in ([] if lines is None else lines[:,0]):
  theta=math.atan2(y2-y1,x2-x1)%math.pi;d=np.array([math.cos(theta),math.sin(theta)]);n=np.array([-d[1],d[0]])
  rho=float(np.array([x1,y1])@n);mid=rho-n@np.array([640.,360.])
  if any(abs((theta-q[0]+math.pi/2)%math.pi-math.pi/2)<.007 and abs(mid-q[1])<5 for q in proposals):continue
  proposals.append((theta,mid,np.array([x1,y1],float),np.array([x2,y2],float)))
 result=[]
 for theta,mid,a,b in proposals:
  d=np.array([math.cos(theta),math.sin(theta)]);n=np.array([-d[1],d[0]])
  # Sample the entire visible axis rather than a fragment picked by Hough.
  rho=float(a@n);uv=np.arange(-1500,1501,5);p=np.array([640.,360.])+n*mid+uv[:,None]*d
  keep=(p[:,0]>14)&(p[:,0]<1266)&(p[:,1]>115)&(p[:,1]<707);p=p[keep]
  if len(p)<25:continue
  offsets=np.arange(-8,9);coords=p[:,None,:]+offsets[None,:,None]*n
  xx=np.rint(coords[:,:,0]).astype(int).clip(0,1279);yy=np.rint(coords[:,:,1]).astype(int).clip(0,719)
  pix=lo[yy,xx];base=np.median(pix[:,[0,1,-2,-1]],axis=1)
  weights=np.maximum(0,pix-base[:,None])*neutral[yy,xx];pk=weights.argmax(1);peak=weights[np.arange(len(p)),pk]
  good=(peak>11)&(abs(offsets[pk])<7)
  widths=(weights>=peak[:,None]*.5).sum(1)
  good&=widths<=12
  # Occluded holds/notes do not invalidate the rest of the line. Reject
  # isolated circle arcs and textured illustration edges by long support.
  if good.sum()<27 or good.mean()<.28:continue
  # A dense grid of vertical rails creates many short crossings on a false
  # horizontal Hough axis. A real rail also has continuous source support.
  changes=np.diff(np.r_[False,good,False].astype(int));runs=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
  if max(runs,default=0)<12 and (runs>=8).sum()<2:continue
  fit=coords[np.arange(len(p)),pk][good].astype(np.float32)
  vx,vy,cx,cy=cv2.fitLine(fit,cv2.DIST_HUBER,0,.005,.005).ravel();d=np.array([float(vx),float(vy)])
  if d@np.array([math.cos(theta),math.sin(theta)])<0:d=-d
  n=np.array([-d[1],d[0]]);rho=float(np.median(fit@n));theta=math.atan2(d[1],d[0])%math.pi
  mid=rho-n@np.array([640.,360.]);width=float(np.clip(np.median(widths[good]),1,30))
  alpha=float(np.clip(np.median(peak[good]/np.maximum(1,255-base[good])),.02,1))
  if any(abs((theta-q['theta']+math.pi/2)%math.pi-math.pi/2)<.008 and abs(mid-q['mid'])<max(5,(width+q['width'])/2) for q in result):continue
  # A target is an endpoint only if the source body occupies ONE side of it.
  # Merely intersecting a ring must not truncate an otherwise full-screen rail.
  anchor=None
  for h in heads:
   c=np.array([h['x'],h['y']]);perp=abs(c@n-rho)
   if perp>5 or h['alpha']<.25:continue
   u=(fit-c)@d;neg=(u<-h['radius']-7).sum();pos=(u>h['radius']+7).sum()
   if min(neg,pos)>max(2,max(neg,pos)*.07):continue
   if max(neg,pos)<20:continue
   if np.min(abs(u))>h['radius']+35:continue
   direction=d if pos>neg else -d
   anchor=c;d=direction;break
  if anchor is None:
   anchor=np.array([640.,360.])+n*mid-d*2300
  result.append({'theta':theta,'mid':mid,'width':width,'alpha':alpha,'anchor':anchor,'rotation':-math.degrees(math.atan2(d[1],d[0])), 'anchored':anchor is not None and -100<anchor[0]<1380 and -100<anchor[1]<820,'score':int(good.sum())})
 return result

def scan(video,root,targets):
 root=Path(root);sig=hashlib.sha256(Path(__file__).read_bytes()+str(Path(video).stat().st_size).encode()+(root/'targets-v2.json').read_bytes()).hexdigest();cache=root/'lines-v6.json'
 if cache.exists():
  obj=json.loads(cache.read_text())
  if obj.get('signature')==sig:return obj['tracks'],obj['report']
 heads={}
 for tr in targets:
  for q in tr['samples']:
   if q['radius']<28:heads.setdefault(round(q['time']*30),[]).append(q)
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);tracks=[];active=[];f=0;count=0
 cv2.setNumThreads(2)
 while True:
  ok,im=cap.read()
  if not ok:break
  ds=detect(im,heads.get(f,[]));t=f/fps;used=set();nxt=[];count+=len(ds)
  if active and ds:
   costs=np.full((len(active),len(ds)),1e6)
   for i,tr in enumerate(active):
    old=tr['_last'];ss=tr['samples'];dt=t-ss[-1]['time'];v=0.;omega=0.
    if len(ss)>2:
     prev=tr['_prev'];dd=max(1/fps,ss[-1]['time']-ss[-2]['time']);v=np.clip((old['mid']-prev['mid'])/dd,-5000,5000);omega=np.clip(((old['theta']-prev['theta']+math.pi/2)%math.pi-math.pi/2)/dd,-12,12)
    for j,q in enumerate(ds):
     angle=abs((q['theta']-(old['theta']+omega*dt)+math.pi/2)%math.pi-math.pi/2);dist=abs(q['mid']-old['mid']-v*dt)
     if angle<.22 and dist<100 and abs(math.log(q['width']/old['width']))<1.2:costs[i,j]=dist+angle*240+abs(q['width']-old['width'])*2
   ii,jj=linear_sum_assignment(costs)
   for i,j in zip(ii,jj):
    if costs[i,j]>=1e6:continue
    tr=active[i];q=ds[j];tr['_prev']=tr['_last'];tr['_last']=q;tr['samples'].append(sample(q,t));nxt.append(tr);used.add(j)
  for j,q in enumerate(ds):
   if j not in used:
    tr={'kind':'rail','samples':[sample(q,t)],'_last':q,'_prev':q};tracks.append(tr);nxt.append(tr)
  # No invented visible bridge over a missing frame: tracks end at each gap.
  active=nxt;f+=1
  if f%600==0:print('V6 all-frame axes',f,count,flush=True)
 cap.release();output=[]
 for tr in tracks:
  ss=tr['samples']
  # Fast source rotations can appear in only one frame at a given axis.
  # Rejecting them erased real lines even with hundreds of ridge pixels.
  if len(ss)<2 and tr['_last']['score']<100:continue
  # Direction matters for finite lines. Unwrap full angles, never modulo180
  # across an on-screen endpoint; source-sidedness is the direction evidence.
  rr=np.degrees(np.unwrap(np.radians([q['rotation'] for q in ss])))
  for q,r in zip(ss,rr):q['rotation']=float(r)
  output.append({'kind':'rail','samples':ss})
 report={'frames':f,'frameAxes':count,'tracks':len(output),'method':'full-resolution source-neutral ridge with continuous support, finite endpoint one-sided support, velocity-predicted rotation','singletonDetectionsRejected':sum(len(tr['samples'])<2 and tr['_last']['score']<100 for tr in tracks)}
 cache.write_text(json.dumps({'signature':sig,'tracks':output,'report':report},separators=(',',':')));(root/'line-review-v6.json').write_text(json.dumps(report,indent=2));return output,report

def sample(q,t):
 return {'time':round(t,8),'x':float((q['anchor'][0]-640)*1.5),'y':float((360-q['anchor'][1])*1.5),'rotation':q['rotation'],'size':q['width']/(1.855675*1.6),'alpha':q['alpha'],'nativeAnchor':True,'anchored':bool(q['anchored'])}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--video',required=True);a=p.parse_args();r=Path(a.root);ts=json.loads((r/'targets-v2.json').read_text());lines,report=scan(a.video,r,ts);old=json.loads((r/'visuals-v2.json').read_text());(r/'visuals-v2.json').write_text(json.dumps(lines+[q for q in old if q['kind']=='target'],separators=(',',':')));print(report)
