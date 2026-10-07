"""Track visible judgement circles and finite rails directly from video pixels."""
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment

def circle_kernel(radius):
 size=int(radius+5)*2+1;yy,xx=np.mgrid[:size,:size];dist=np.hypot(xx-size//2,yy-size//2)
 ring=np.exp(-((dist-radius)/.72)**2);inner=(dist<radius-3).astype(float);outer=((dist>radius+2)&(dist<radius+4)).astype(float)
 return (ring/ring.sum()-.65*inner/inner.sum()-.35*outer/outer.sum()).astype(np.float32)

def remove_hud_glyphs(tracks):
 out=[]
 for tr in tracks:
  parts=[]
  for q in tr['samples']:
   x,y=q['x'],q['y']
   if 20<y<112 and (x<310 or 550<x<750 or x>1060):continue
   if not parts or q['time']-parts[-1][-1]['time']>.35:parts.append([])
   parts[-1].append(q)
  for i,part in enumerate(parts):
   if len(part)>=4:out.append({**tr,'id':tr['id']+i*1000000,'samples':part})
 return out

def scan(video,root,rails_only=False):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30;f=0;circles=[];rails=[];active=[];ractive=[];kernel=circle_kernel(10.3);ghostkernel=circle_kernel(19.5)
 def attach(detections,active,tracks,frame,maxdist,kind):
  nxt=[];used=set()
  if active and detections:
   cost=np.ones((len(active),len(detections)))*1e6
   for i,tr in enumerate(active):
    last=tr['samples'][-1];dt=frame-last['time'];px,py=last['x'],last['y']
    if len(tr['samples'])>2:
     prev=tr['samples'][-3];dd=last['time']-prev['time']
     if dd>0:px+=np.clip((last['x']-prev['x'])/dd,-900,900)*dt;py+=np.clip((last['y']-prev['y'])/dd,-900,900)*dt
    for j,d in enumerate(detections):
     dist=math.hypot(d['x']-px,d['y']-py)
     if abs(d.get('radius',0)-last.get('radius',0))>8:continue
     if kind=='rail':
      angle=abs((d['rotation']-last['rotation']+90)%180-90)
      if angle>12:continue
      dist+=angle*4
     if dist<maxdist+dt*500:cost[i,j]=dist
   ii,jj=linear_sum_assignment(cost)
   for i,j in zip(ii,jj):
    if cost[i,j]>=1e6:continue
    tr=active[i];tr['samples'].append(detections[j]);nxt.append(tr);used.add(j)
  for j,d in enumerate(detections):
   if j in used:continue
   tr={'id':len(tracks),'kind':kind,'samples':[d]};tracks.append(tr);nxt.append(tr)
  for tr in active:
   if not any(tr is q for q in nxt) and frame-tr['samples'][-1]['time']<(.32 if kind=='target' else .12):nxt.append(tr)
  return nxt
 while True:
  ok,raw=cap.read()
  if not ok:break
  im=cv2.resize(raw,(640,360),interpolation=cv2.INTER_AREA).astype(np.float32);t=f/fps;lo=im.min(2);chroma=im.max(2)-lo
  white=lo*np.clip(1-(chroma-10)/40,0,1);white[:10,:]=0
  detections=[]
  for radius,k,minscore in (() if rails_only else ((10.3,kernel,14),(19.5,ghostkernel,8))):
   score=cv2.filter2D(white,-1,k);maximum=cv2.dilate(score,np.ones((9,9),np.uint8));ys,xs=np.nonzero((score>=maximum-.001)&(score>minscore))
   for y,x in zip(ys,xs):
    if y<13 or y>349 or x<3 or x>636:continue
    values=[]
    for angle in np.arange(32)*math.pi/16:
     sy=int(round(y+radius*math.sin(angle)));sx=int(round(x+radius*math.cos(angle)))
     if 1<=sy<359 and 1<=sx<639:values.append(float(white[sy-1:sy+2,sx-1:sx+2].max()))
    if len(values)<24:continue
    center=float(np.median(lo[max(0,y-4):y+5,max(0,x-4):x+5]));contrast=np.array(values)-center
    if radius<15 and center>110:continue
    if (contrast>max(12,minscore)).mean()<.69:continue
    if radius<15 and np.median(values)<65:continue
    if any(math.hypot(x-d['x']/2,y-d['y']/2)<10 for d in detections):continue
    detections.append({'time':round(t,5),'x':float(x*2),'y':float(y*2),'radius':radius*2,'alpha':round(float(np.clip(np.median(contrast)/220,0,1)),4),'score':round(float(score[y,x]),2)})
  active=attach(detections,active,circles,t,24,'target')
  contrast=cv2.morphologyEx(lo,cv2.MORPH_TOPHAT,np.ones((7,7),np.uint8))
  mask=((lo>45)&(chroma<28)&(contrast>13)).astype(np.uint8)*255;mask[:55]=0
  found=cv2.HoughLinesP(mask,1,np.pi/720,threshold=45,minLineLength=90,maxLineGap=24)
  ds=[]
  for line in ([] if found is None else found[:,0]):
   x1,y1,x2,y2=map(float,line);theta=math.atan2(y2-y1,x2-x1)%math.pi;nx,ny=-math.sin(theta),math.cos(theta);rho=x1*nx+y1*ny
   foundq=None
   for q in ds:
    if abs((theta-q['theta']+math.pi/2)%math.pi-math.pi/2)<.014 and abs(rho-q['rho'])<3:foundq=q;break
   if foundq:foundq['ends'].extend([(x1,y1),(x2,y2)]);continue
   ds.append({'theta':theta,'rho':rho,'ends':[(x1,y1),(x2,y2)]})
  rd=[]
  for q in ds:
   theta=q['theta'];dx,dy=math.cos(theta),math.sin(theta);nx,ny=-dy,dx;rho=q['rho'];projections=[x*dx+y*dy for x,y in q['ends']];u,v=min(projections),max(projections)
   x1,y1=u*dx+rho*nx,u*dy+rho*ny;x2,y2=v*dx+rho*nx,v*dy+rho*ny
   if abs(dy)>.05:
    if 53<y1<60:x1+=(0-y1)*dx/dy;y1=0
    if 53<y2<60:x2+=(0-y2)*dx/dy;y2=0
   cx,cy=(x1+x2)/2,(y1+y2)/2;length=math.hypot(x2-x1,y2-y1)
   values=[]
   for frac in np.linspace(.05,.95,50):
    px=int(np.clip((x1+(x2-x1)*frac)/1,3,636));py=int(np.clip(y1+(y2-y1)*frac,3,356));values.append(float(lo[py-1:py+2,px-1:px+2].max()-np.median(lo[py-3:py+4,px-3:px+4])))
   alpha=float(np.clip(np.median(values)/220,.05,1))
   rd.append({'time':round(t,5),'x':cx*2,'y':cy*2,'rotation':theta*180/math.pi,'width':length*3/512,'alpha':alpha,'ends':[x1*2,y1*2,x2*2,y2*2]})
  ractive=attach(rd,ractive,rails,t,65,'rail');f+=1
  if f%900==0:print(f'geometry: {f} frames',flush=True)
 cap.release()
 circles=remove_hud_glyphs([q for q in circles if len(q['samples'])>=4]);rails=[q for q in rails if len(q['samples'])>=3]
 if not rails_only:(root/'targets-v2.json').write_text(json.dumps(circles,separators=(',',':')))
 (root/'rails-v2.json').write_text(json.dumps(rails,separators=(',',':')))
 summary={'frames':f,'fps':fps,'targetTracks':len(circles),'railTracks':len(rails),'targetSamples':sum(len(q['samples']) for q in circles)}
 (root/'geometry-report.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',required=True);p.add_argument('--rails-only',action='store_true');a=p.parse_args();scan(a.video,a.output,a.rails_only)
