"""Refit native line-head rings; reject filled bodies and unsupported gap bridges."""
import json,hashlib,inspect,copy
from pathlib import Path
import cv2,numpy as np

PHI=np.arange(64)*np.pi/32
UNIT=np.column_stack([np.cos(PHI),np.sin(PHI)])

def measure(im,q,planes=None):
 if planes is None:lo=im.min(2).astype(float);neutral=(im.max(2).astype(float)-lo)<29
 else:lo,neutral=planes
 center=np.array([q['x'],q['y']]);r0=q['radius']
 radii=np.arange(max(9,r0*.65),min(100,r0*1.9)+.1,1.)
 def profiles(c):
  coords=c+radii[:,None,None]*UNIT
  vals=[]
  for offset in [-2,-1,0,1,2]:
   pts=coords+offset*UNIT;xx=np.rint(pts[:,:,0]).astype(int);yy=np.rint(pts[:,:,1]).astype(int)
   valid=(xx>=0)&(xx<1280)&(yy>=112)&(yy<716)
   vals.append(np.where(valid&neutral[yy.clip(0,719),xx.clip(0,1279)],lo[yy.clip(0,719),xx.clip(0,1279)],0))
  peaks=np.max(vals,axis=0);bases=[]
  for offset in [-6,6]:
   pts=coords+offset*UNIT;xx=np.rint(pts[:,:,0]).astype(int).clip(0,1279);yy=np.rint(pts[:,:,1]).astype(int).clip(0,719)
   bases.append(lo[yy,xx])
  # A bright vertical capsule is not a ring. The ring must rise above BOTH
  # the inside and outside local backgrounds around most angular sectors.
  excess=peaks-np.maximum(*bases);good=(excess>8)&(peaks>35)
  valid=(coords[:,:,0]>=0)&(coords[:,:,0]<1280)&(coords[:,:,1]>=112)&(coords[:,:,1]<716)
  coverage=(good&valid).sum(1)/np.maximum(1,valid.sum(1))
  scores=np.percentile(excess,35,axis=1)+coverage*40-abs(radii-r0)*.04
  scores[(coverage<.58)|(valid.sum(1)<40)]=-1e6
  k=int(np.argmax(scores));return float(scores[k]),k,peaks[k],excess[k],good[k]
 score,k,peak,excess,good=profiles(center)
 # Small detector/smoothing offsets can move a two-pixel circle off its ridge.
 if score<30:
  for dx,dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]:
   c=np.array([q['x']+dx,q['y']+dy]);result=profiles(c)
   if result[0]>score:score,k,peak,excess,good=result;center=c
 if score<-1e5:return None
 alpha=float(np.clip(np.median(excess[good])/220,.02,1.))
 return {**q,'x':float(center[0]),'y':float(center[1]),'radius':float(radii[k]),'alpha':alpha}

def refine(video,root,effects):
 root=Path(root);targets=[q for q in effects if q['kind']=='target'];cache=root/'targets-reviewed-v6.json'
 signature=hashlib.sha256(Path(__file__).read_bytes()+json.dumps(targets,separators=(',',':')).encode()+str(Path(video).stat().st_size).encode()).hexdigest()
 if cache.exists():
  old=json.loads(cache.read_text())
  if old.get('signature')==signature:return [q for q in effects if q['kind']!='target']+old['tracks'],old['report']
 byframe={};measured={};observations=0;removed=0;resized=0
 for i,tr in enumerate(targets):
  for q in tr['samples']:byframe.setdefault(round(q['time']*30),[]).append((i,q))
 cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);f=0;cv2.setNumThreads(2)
 while True:
  ok,im=cap.read()
  if not ok:break
  qs=byframe.get(f,[])
  lo=im.min(2).astype(float) if qs else None
  planes=(lo,(im.max(2).astype(float)-lo)<29) if qs else None
  for i,q in qs:
   observations+=1;v=measure(im,q,planes)
   if v is None:removed+=1;continue
   resized+=abs(v['radius']-q['radius'])>2
   measured.setdefault(i,[]).append({**v,'time':f/fps})
  f+=1
  if f%900==0:print('V6 native ring validation',f,flush=True)
 cap.release();output=[]
 for i,ss in measured.items():
  parts=[]
  for q in ss:
   if not parts or q['time']-parts[-1][-1]['time']>1.5/fps:parts.append([])
   parts[-1].append(q)
  for p in parts:output.append({**targets[i],'samples':p})
 report={'sourceSamples':observations,'unsupportedRingSamplesRemoved':removed,'radiusSamplesRefitted':int(resized),'sourceContinuousFragments':len(output)}
 cache.write_text(json.dumps({'signature':signature,'tracks':output,'report':report},separators=(',',':')))
 return [q for q in effects if q['kind']!='target']+output,report
