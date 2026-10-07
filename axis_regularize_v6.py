"""Remove ridge-fitting angle noise only in strongly parallel dense arrays."""
import copy,math
import numpy as np

def regularize(effects):
 out=copy.deepcopy(effects or []);frames={};changed=0
 for tr in out:
  if tr.get('kind')!='rail':continue
  previous=None
  for q in tr['samples']:
   if previous is not None and not q.get('anchored') and not previous.get('anchored'):
    old=q['rotation'];turn=round((previous['rotation']-old)/180)
    if turn%2:
     a=math.radians(-old);q['x']+=math.cos(a)*4600*1.5;q['y']-=math.sin(a)*4600*1.5
    q['rotation']=old+turn*180
   previous=q
  for q in tr['samples']:frames.setdefault(round(q['time']*30),[]).append(q)
 for ss in frames.values():
  for axis in (0,90):
   family=[q for q in ss if abs((q['rotation']-axis+90)%180-90)<1.5]
   if len(family)<6:continue
   residual=np.array([(q['rotation']-axis+90)%180-90 for q in family]);median=float(np.median(residual))
   # Three separate tilts must remain separate. A six-line array with a
   # tightly supported common axis can reject occasional ring-edge outliers.
   deviation=float(np.median(abs(residual-median)))
   if not ((deviation<=.12 and abs(median)<=.18) or (len(family)>=12 and deviation<=.45 and abs(median)<=.3)):continue
   for q in family:
    old=q['rotation'];new=old-((old-axis+90)%180-90)
    if abs(old-new)<.005:continue
    if not q.get('anchored'):
     a=math.radians(-old);b=math.radians(-new)
     q['x']+=(math.cos(a)-math.cos(b))*2300*1.5
     q['y']-=(math.sin(a)-math.sin(b))*2300*1.5
    q['rotation']=new;changed+=1
 return out,{'denseParallelAxisSamplesRegularized':changed}
