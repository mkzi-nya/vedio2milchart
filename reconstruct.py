"""Video-only reconstruction. Never opens any source chart/archive folder."""
import subprocess,json,csv,argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='从视频追踪 Milthm 音符，不读取原谱')
parser.add_argument('video');parser.add_argument('--output',default=str(ROOT/'output'));parser.add_argument('--judge-y',type=float,default=.82361);parser.add_argument('--title',default='Video reconstruction');parser.add_argument('--threshold',type=int,default=168);parser.add_argument('--min-frames',type=int,default=4)
args=parser.parse_args();VIDEO=str(Path(args.video).resolve());ROOT=Path(args.output).resolve();ROOT.mkdir(parents=True,exist_ok=True)
JUDGE=args.judge_y*360
cmd=['ffmpeg','-v','error','-i',VIDEO,'-vf','fps=30:start_time=0:round=up,scale=640:360','-f','rawvideo','-pix_fmt','rgb24','-']
p=subprocess.Popen(cmd,stdout=subprocess.PIPE)
tracks=[]; active=[]; frame=0; targets=[]
while True:
 raw=p.stdout.read(640*360*3)
 if len(raw)!=640*360*3:break
 im=np.frombuffer(raw,np.uint8).reshape(360,640,3)
 # Notes are pastel white; purple hit particles fail the red/green floor.
 mask=(im[:,:,0]>160)&(im[:,:,1]>args.threshold)&(im[:,:,2]>175)
 mask[:48,:]=False; mask[:60,:145]=False; mask[:60,540:]=False; mask[min(359,int(JUDGE+11)):,:]=False
 # Detect thin white judgement circles before removing one-pixel rail lines.
 white=(im.min(axis=2)>175)&((im.max(axis=2).astype(int)-im.min(axis=2).astype(int))<20)
 white[:50,:]=False;white[335:,:]=False
 tl,tn=ndimage.label(white)
 for ti,ts in enumerate(ndimage.find_objects(tl)):
  if ts is None:continue
  ty,tx=ts;tw=tx.stop-tx.start;th=ty.stop-ty.start
  if not (18<=tw<=28 and 18<=th<=28 and .9<tw/th<1.12):continue
  ta=np.count_nonzero(tl[ts]==ti+1)
  if .12<ta/(tw*th)<.55:targets.append([frame/30,(tx.start+tx.stop-1)/2,(ty.start+ty.stop-1)/2])
 mask=ndimage.binary_opening(mask,structure=np.ones((2,2)))
 labels,n=ndimage.label(mask)
 objs=ndimage.find_objects(labels); det=[]
 for idx,sl in enumerate(objs):
  if sl is None:continue
  yy,xx=sl; w=xx.stop-xx.start; h=yy.stop-yy.start
  area=np.count_nonzero(labels[sl]==idx+1)
  if not (10<=w<=28 and 10<=h<=300 and area>=40):continue
  if h<35 and not (.45<w/h<1.7):continue
  if h>35 and not (w<=25):continue
  x=(xx.start+xx.stop-1)/2; y=(yy.start+yy.stop-1)/2
  hold=h>35
  if hold:y=yy.stop-1-w/2
  fill=area/(w*h)
  pix=im[sl][labels[sl]==idx+1].astype(float)
  # White round outlines are judgement targets; pastel teardrops are Drag notes.
  if not hold and .90<w/h<1.12 and .12<fill<.60 and np.mean(np.max(pix,axis=1)-np.min(pix,axis=1))<20:
   targets.append([frame/30,x,y]);continue
  typ=0 if hold or fill>.60 else 1
  det.append([frame/30,x,y,w,h,typ,hold,fill,yy.start])
 used=set();new=[]
 for d in det:
  best=None;cost=1e9
  for tr in active:
   if tr['id'] in used:continue
   last=tr['samples'][-1];dt=d[0]-last[0]
   if dt>.13:continue
   v=0 if last[6] and last[2]>280 else 230
   if len(tr['samples'])>=3:
    prev=tr['samples'][-3];v=np.clip((last[2]-prev[2])/(last[0]-prev[0]),-100,500)
   dx=d[1]-last[1];dy=d[2]-(last[2]+v*dt)
   c=dx*dx+dy*dy
   if abs(dx)<18 and abs(dy)<25 and c<cost:best=tr;cost=c
  if best is None:
   best={'id':len(tracks),'samples':[]};tracks.append(best)
  best['samples'].append(d);used.add(best['id']);new.append(best)
 active=new+[tr for tr in active if tr['id'] not in used and frame/30-tr['samples'][-1][0]<.10]
 frame+=1
p.wait()
events=[]
targets=np.array(targets).reshape(-1,3)
for tr in tracks:
 s=np.array(tr['samples']);s=s[s[:,2]<JUDGE-6]
 if len(s)<args.min_frames:continue
 # Fit falling head away from judgement particles.
 tail=s[-min(12,len(s)):];v,inter=np.polyfit(tail[:,0],tail[:,2],1)
 vx,ix=np.polyfit(tail[:,0],tail[:,1],1)
 velocity=np.hypot(v,vx)
 if velocity<50 or velocity>650:continue
 hit=(JUDGE-inter)/v if abs(v)>50 else s[-1,0]+10
 default_valid=s[-1,2]>=JUDGE-77 and s[-1,0]-.1<hit<s[-1,0]+.65
 x=float(vx*hit+ix);jy=JUDGE;moving=False
 # Find a visible target near the projected trajectory, including raised targets.
 nearby=targets[np.abs(targets[:,0]-s[-1,0])<.11]
 choices=[]
 for _,tx,ty in nearby:
  dt=((tx-s[-1,1])*vx+(ty-s[-1,2])*v)/(velocity**2)
  distance=np.hypot(s[-1,1]+vx*dt-tx,s[-1,2]+v*dt-ty)
  if -.08<dt<.65 and distance<15:
   choices.append((abs(dt),s[-1,0]+dt,tx,ty))
 if choices:
  _,candidate,cx,cy=min(choices)
  if not default_valid or abs(candidate-hit)<.45:
   hit=float(candidate);x=float(cx);jy=float(cy);moving=abs(jy-JUDGE)>8
 if not choices and not default_valid:continue
 if not 0<=hit<frame/30:continue
 held=s[:,6].sum()>=3
 duration=float(np.median((s[s[:,6]>0,4]-s[s[:,6]>0,3])/velocity)) if held else 0
 typ=0 if held else int(np.median(s[:,5])>=.5)
 ev={'time':round(float(hit),5),'x':round(x*2,2),'type':typ,'duration':round(max(0,duration),5),'track':tr['id'],'frames':len(s),'speed':round(float(velocity*2),2),'confidence':'medium' if len(s)>8 else 'low','sourceTime':round(float(hit),5),'sourceX':round(x*2,2),'judgeY':round(float(jy/360),6),'movingTarget':moving,'path':[[round(float(q[0]),5),round(float(q[1]*2),2),round(float(q[2]*2),2)] for q in s[::2]]}
 if 0<=hit<frame/30:events.append(ev)
events.sort(key=lambda e:e['time'])
# Remove fragmentation duplicates without inventing unseen notes.
clean=[]
for e in events:
 if any(abs(e['time']-q['time'])<.075 and abs(e['x']-q['x'])<28 for q in clean[-15:]):continue
 clean.append(e)
(ROOT/'events.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2))
(ROOT/'targets.json').write_text(json.dumps(targets.tolist(),separators=(',',':')))
(ROOT/'tracks.json').write_text(json.dumps(tracks,separators=(',',':')))
with (ROOT/'events.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['time','x','type','duration','track','frames','speed','confidence','sourceTime','sourceX','judgeY','movingTarget','path']);w.writeheader();w.writerows(clean)
from export import export_js
(ROOT/'chart.js').write_text(export_js(clean,args.title,args.judge_y),encoding='utf-8')
report={'frames':frame,'duration':frame/30,'tracks':len(tracks),'notes':len(clean),'holds':sum(e['duration']>0 for e in clean),'drags':sum(e['type']==1 for e in clean),'method':'pastel component tracking + local linear judgement extrapolation','limitations':['遮挡可能漏识别或产生重复','长条长度为像素估计','移动判定圆仅在可见且未被特效遮挡时估计；旋转轨道需要校正','未恢复 AP、假音符、故事板和原始 BPM'],'source':VIDEO}
(ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
