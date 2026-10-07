"""Stable video-observed heads and conservative dense combo anchors."""
import cv2,numpy as np,json,math
from pathlib import Path

def fixed_heads(video,root):
 root=Path(root);tracks=json.loads((root/'targets-v2.json').read_text());points=[s for tr in tracks for s in tr['samples'] if s['radius']<28 and abs(s['y']-592)<4]
 hist=np.bincount([round(q['x']/4) for q in points],minlength=321);peaks=[]
 for k in np.argsort(hist)[::-1]:
  if hist[k]<180:break
  if all(abs(k-q)>12 for q in peaks):peaks.append(int(k))
 xs=sorted(float(np.median([q['x'] for q in points if abs(q['x']-k*4)<8])) for k in peaks);cap=cv2.VideoCapture(str(video));frame=0;ss=[[] for _ in xs];angles=np.arange(40)*math.pi/20
 while True:
  ok,im=cap.read()
  if not ok:break
  small=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA).astype(float);lo=small.min(2);ch=small.max(2)-lo
  for i,x in enumerate(xs):
   cx=x/2;vals=[];background=[]
   for angle in angles:
    sy=int(round(296+10.3*math.sin(angle)));sx=int(round(cx+10.3*math.cos(angle)));pixel=lo[sy-1:sy+2,sx-1:sx+2]*np.clip(1-(ch[sy-1:sy+2,sx-1:sx+2]-10)/45,0,1);vals.append(pixel.max());sy2=int(round(296+15*math.sin(angle)));sx2=int(round(cx+15*math.cos(angle)));background.append(lo[sy2,sx2])
   contrast=max(0,np.percentile(vals,65)-np.median(background));alpha=float(np.clip(contrast/220,0,1));alpha=alpha if (np.asarray(vals)>65).mean()>.4 else 0
   ss[i].append({'time':round(frame/30,5),'x':x,'y':592.,'radius':20.6,'alpha':round(alpha,3)})
  frame+=1
 cap.release();out=[{'id':-i-1,'kind':'target','samples':s,'evidence':'fixed-head-template'} for i,s in enumerate(ss)]
 # Suppress short particle lookalikes around permanent heads, preserving raised targets.
 for tr in tracks:
  s=tr['samples'];isfixed=np.mean([q['radius']<28 and abs(q['y']-592)<38 and min([abs(q['x']-x) for x in xs] or [999])<40 for q in s])>.6
  if not isfixed and (len(s)>=7 or s[0]['radius']>28):out.append(tr)
 (root/'heads-clean.json').write_text(json.dumps(out,separators=(',',':')));print('fixed heads',xs,'clean tracks',len(out),flush=True);return out,xs

def dense_combo(combo):
 sparse=[(0.,0)]
 for t,c in combo['ocrLabels']:
  if t>sparse[-1][0] and c>=sparse[-1][1] and c<=combo['finalCombo'] and c-sparse[-1][1]<=(t-sparse[-1][0])*40+10:sparse.append((t,c))
 candidates=[];samples=combo['samples']
 for i,q in enumerate(samples):
  t,c=q['time'],q['combo'];left=max((a for a in sparse if a[0]<=t),default=sparse[0]);right=min((a for a in sparse if a[0]>=t),default=sparse[-1])
  if not left[1]<=c<=right[1]:continue
  if q['error']>.045:continue
  if i and c-samples[i-1]['combo']>max(8,(t-samples[i-1]['time'])*30):continue
  candidates.append((t,c))
 dense=list(sparse)
 for t in np.arange(1,sparse[-1][0],1):
  close=[q for q in candidates if abs(q[0]-t)<.18]
  if close:dense.append(min(close,key=lambda q:abs(q[0]-t)))
 # A long unchanged observed counter supplies plateau constraints for fake-note passages.
 for q,nxt in zip(samples,samples[1:]):
  if nxt['time']-q['time']>1 and q['combo']<=combo['finalCombo']:
   if any(c==q['combo'] and q['time']-.2<=t<=nxt['time']+.2 for t,c in sparse):
    for t in np.arange(q['time']+.4,nxt['time']-.2,.5):dense.append((float(t),q['combo']))
 dense.sort();out=[]
 for t,c in dense:
  if not out or t>out[-1][0]+.08 and c>=out[-1][1]:out.append((float(t),int(c)))
 return {**combo,'ocrLabels':out}
