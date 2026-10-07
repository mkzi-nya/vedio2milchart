"""Video-only line geometry and purple judgement effect evidence."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy import ndimage,signal
P=argparse.ArgumentParser();P.add_argument('video');P.add_argument('--output',required=True);a=P.parse_args();root=Path(a.output)
p=subprocess.Popen(['ffmpeg','-v','error','-i',a.video,'-vf','fps=15:start_time=0:round=up,scale=320:180','-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
angles=np.arange(0,180,2)*np.pi/180;ca=np.cos(angles);sa=np.sin(angles);radius=370;active=[];lines=[];purple=[];f=0
while True:
 raw=p.stdout.read(320*180*3)
 if len(raw)!=320*180*3:break
 im=np.frombuffer(raw,np.uint8).reshape(180,320,3).astype(float)
 white=(im.min(axis=2)>170)&(im.max(axis=2)-im.min(axis=2)<25);white[:28,:]=False;white[166:]=False
 yy,xx=np.nonzero(white);acc=np.zeros((90,radius*2),np.int32)
 if len(xx):
  for k in range(90):
   bins=np.rint(xx*ca[k]+yy*sa[k]).astype(int)+radius
   acc[k]=np.bincount(bins,minlength=radius*2)[:radius*2]
 peaks=(acc==ndimage.maximum_filter(acc,size=(5,9)))&(acc>65)
 detections=[]
 for k,r in zip(*np.nonzero(peaks)):
  theta=angles[k];rho=r-radius
  # Extend observed line through the screen rectangle, then retain visible geometry.
  ends=[]
  for x in (0,320):
   if abs(sa[k])>.001:
    y=(rho-x*ca[k])/sa[k]
    if 28<=y<=166:ends.append((x,y))
  for y in (28,166):
   if abs(ca[k])>.001:
    x=(rho-y*sa[k])/ca[k]
    if 0<=x<=320:ends.append((x,y))
  if len(ends)<2:continue
  u,v=ends[:2];length=np.hypot(v[0]-u[0],v[1]-u[1]);cx=(u[0]+v[0])/2;cy=(u[1]+v[1])/2
  detections.append({'time':f/15,'x':(cx/320-.5)*1920,'y':(.5-cy/180)*1080,'rotation':float(np.degrees(np.arctan2(v[1]-u[1],v[0]-u[0]))),'width':float(length*6/512),'angle':int(k),'rho':int(rho)})
 used=set();nxt=[]
 for d in detections:
  options=[(abs(d['rho']-q['samples'][-1]['rho'])+abs(d['angle']-q['samples'][-1]['angle'])*3,q) for q in active if q['id'] not in used and abs(d['angle']-q['samples'][-1]['angle'])<5 and abs(d['rho']-q['samples'][-1]['rho'])<22]
  q=min(options,key=lambda q:q[0])[1] if options else {'id':len(lines),'samples':[]}
  if not options:lines.append(q)
  q['samples'].append(d);used.add(q['id']);nxt.append(q)
 active=nxt
 # Regional purple energy: a hit ring must expand around an existing judgement area.
 mask=(im[:,:,2]-im[:,:,1]>28)&(im[:,:,0]>85)&(im[:,:,2]>150)
 weights=ndimage.gaussian_filter(mask.astype(float),5)
 maxima=(weights==ndimage.maximum_filter(weights,size=21))&(weights>.075)
 coords=list(zip(*np.nonzero(maxima)))
 purple.append({'time':f/15,'centers':[[float(x*4),float(y*4),float(weights[y,x])] for y,x in coords if y>30]})
 f+=1
p.wait()
lines=[q for q in lines if len(q['samples'])>=4]
(root/'effects.json').write_text(json.dumps(lines,separators=(',',':')))
(root/'hit-evidence.json').write_text(json.dumps(purple,separators=(',',':')))
print(json.dumps({'lineTracks':len(lines),'effectFrames':len(purple)},ensure_ascii=False))
