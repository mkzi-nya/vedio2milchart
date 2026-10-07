"""Locate hold head contact from the observed deceleration into a moving target."""
import numpy as np

def fit_holds(events):
 changed=0
 for e in events:
  if e.get('duration',0)<=0 or e.get('isFake') or not e.get('tailPath') or abs(e.get('rotation',90)-90)>5:continue
  p=np.asarray(e.get('path',[]),float)
  if len(p)<7:continue
  v=np.diff(p[:,2])/np.maximum(.001,np.diff(p[:,0]));which=[]
  for j in range(2,len(p)-2):
   if abs(v[j])<160 and np.median(v[max(0,j-3):j])>400 and p[j,2]>250:which.append(j)
  if not which:continue
  j=which[0];velocity=float(np.median(v[max(0,j-3):max(1,j-1)]));y=float(np.median(p[j:j+3,2]));t=float(p[j-1,0]+(y-p[j-1,2])/max(300,velocity));t=float(np.clip(t,p[j-1,0],p[j,0]+.015))
  if abs(t-e['time'])>.5:continue
  end=e['time']+e['duration'];x=float(np.interp(t,p[:,0],p[:,1]));e.update(time=round(t,5),sourceTime=round(t,5),x=round(x,3),sourceX=round(x,3),judgeY=y/720,duration=round(max(.03,end-t),5),holdEvidence='head-deceleration-and-observed-tail');changed+=1
 return changed
