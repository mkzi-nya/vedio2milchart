"""Constrain real/fake hypotheses against consistent video OCR anchors."""
def refine(events,combo):
 anchors=[(0.,0)]
 for t,c in combo.get('ocrLabels',[]):
  prev_t,prev_c=anchors[-1]
  if t>prev_t and c>=prev_c and c-prev_c<=(t-prev_t)*40+10:anchors.append((float(t),int(c)))
 final=combo.get('finalCombo',0)
 if combo.get('samples'):
  t=combo['samples'][-1]['time']
  if t>anchors[-1][0] and final>=anchors[-1][1]:anchors.append((float(t),int(final)))
 report=[]
 def inside(e,start,end):
  return int(start+.035<e['time']<=end+.035)+int(e['duration']>0 and start+.035<e['time']+e['duration']<=end+.035)
 for (start,prev),(end,nxt) in zip(anchors,anchors[1:]):
  expected=nxt-prev
  relevant=[e for e in events if inside(e,start,end)>0]
  actual=sum(inside(e,start,end) for e in relevant if not e.get('isFake',False))
  if actual>expected:
   for e in sorted((e for e in relevant if not e.get('isFake',False)),key=lambda e:(e.get('frames',0),e['duration']>0)):
    w=inside(e,start,end)
    if actual-w<expected:continue
    e['isFake']=True;e['confidence']='inferred';e['evidence']='visible-track-without-local-counter';actual-=w
    if actual<=expected:break
  elif actual<expected:
   candidates=[e for e in relevant if e.get('isFake',False) and e.get('evidence')!='decorative-track-hypothesis']
   for e in sorted(candidates,key=lambda e:e.get('frames',0),reverse=True):
    w=inside(e,start,end)
    if actual+w>expected:continue
    e['isFake']=False;e['confidence']='inferred';e['evidence']='local-counter-and-visible-track';actual+=w
    if actual>=expected:break
  report.append({'from':start,'to':end,'videoJudgements':expected,'estimatedJudgements':actual,'difference':expected-actual})
 return report

def repair(events,intervals,hitframes,segments,judge):
 repairs=[]
 for interval in intervals:
  gap=interval['difference']
  if gap<=0:continue
  points=[]
  for frame in hitframes:
   t=frame['time']-.065
   if not interval['from']+.035<t<=interval['to']+.035:continue
   for x,y,power in frame['centers']:
    if abs(y-judge*720)<90 and power>.065:points.append((t,x,y,power))
  for t,x,y,power in sorted(points,key=lambda q:q[3],reverse=True):
   if any(abs(t-e['time'])<.095 and abs(x-e['x'])<65 or e['duration']>0 and abs(t-e['time']-e['duration'])<.095 and abs(x-e['x'])<65 for e in events if not e.get('isFake',False)):continue
   if any(abs(t-e['time'])<.13 and abs(x-e['x'])<70 for e in repairs):continue
   if any(0<t-q[0]<.13 and abs(x-q[1])<55 and q[3]>power*.85 for q in points):continue
   nearby=min(events,key=lambda e:abs(e['time']-t)*35+abs(e['x']-x)) if events else None
   e={'time':round(t,5),'sourceTime':round(t,5),'x':x,'sourceX':x,'judgeY':judge,'duration':0,'type':nearby['type'] if nearby else 0,'frames':0,'speed':segments[min(len(segments)-1,int(t//8))]['pixelsPerSecondAt1280'],'confidence':'inferred','isFake':False,'evidence':'hit-effect-and-local-counter','path':[]}
   events.append(e);repairs.append(e);gap-=1
   if gap<=0:break
 return repairs

def settle(events,intervals,time_limit=15):
 """Solve all interval budgets together, including holds crossing boundaries."""
 import numpy as np
 from scipy.optimize import milp,Bounds,LinearConstraint
 from scipy.sparse import csc_matrix
 if not intervals:return {'state':'no-anchors'}
 matrix=np.zeros((len(intervals)+1,len(events)))
 for i,q in enumerate(intervals):
  for j,e in enumerate(events):matrix[i,j]=int(q['from']+.035<e['time']<=q['to']+.035)+int(e['duration']>0 and q['from']+.035<e['time']+e['duration']<=q['to']+.035)
 matrix[-1]=[1+(e['duration']>0) for e in events]
 expected=np.array([q['videoJudgements'] for q in intervals]+[sum(q['videoJudgements'] for q in intervals)])
 scores=-np.array([np.log1p(e.get('frames',0))+.3*(e.get('confidence')=='medium')+2.5*(e.get('evidence')=='dynamic-target-intersection')-2*(e.get('evidence')=='full-screen-decorative-track') for e in events])
 upper=np.array([0 if e.get('evidence') in ('decorative-track-hypothesis','full-screen-decorative-track') else 1 for e in events])
 result=milp(scores,integrality=np.ones(len(events)),bounds=Bounds(0,upper),constraints=LinearConstraint(csc_matrix(matrix),expected,expected),options={'time_limit':time_limit})
 if result.x is None:return {'state':'unresolved','solverStatus':int(result.status),'message':result.message}
 selected=np.rint(result.x).astype(int)
 if not np.array_equal(matrix@selected,expected):return {'state':'unresolved','message':'no feasible exact interval allocation'}
 for e,real in zip(events,selected):
  fake=not bool(real)
  if fake!=e.get('isFake',False):e['confidence']='inferred';e['evidence']='joint-counter-and-track-hypothesis'
  e['isFake']=fake
 for i,q in enumerate(intervals):q['estimatedJudgements']=int((matrix@selected)[i]);q['difference']=0
 return {'state':'feasible-count-allocation','solverStatus':int(result.status),'note':'这是同时满足连击计数的推断，不证明逐个音符的真假属性。'}

def settle_with_frame_uncertainty(events,intervals,time_limit=30):
 """Allow one video frame of judgement uncertainty, retaining observed pixel paths."""
 import numpy as np
 from scipy.optimize import milp,Bounds,LinearConstraint
 from scipy.sparse import coo_matrix
 n=len(events);m=len(intervals);rows=[];cols=[];values=[];options=[];costs=[]
 for j,e in enumerate(events):
  for shift in (-1/30,0,1/30):
   t=max(0,e['time']+shift);end=t+e['duration'];col=len(options);options.append((j,shift));rows.append(j);cols.append(col);values.append(1)
   for i,q in enumerate(intervals):
    weight=int(q['from']+.035<t<=q['to']+.035)+int(e['duration']>0 and q['from']+.035<end<=q['to']+.035)
    if weight:rows.append(n+i);cols.append(col);values.append(weight)
   rows.append(n+m);cols.append(col);values.append(1+(e['duration']>0))
   quality=np.log1p(e.get('frames',0))+2.5*(e.get('targetTrack') is not None)-2*(e.get('evidence')=='full-screen-decorative-track')
   costs.append(-quality+abs(shift)*24)
 matrix=coo_matrix((values,(rows,cols)),shape=(n+m+1,len(options))).tocsc();expected=np.array([q['videoJudgements'] for q in intervals]+[sum(q['videoJudgements'] for q in intervals)])
 lower=np.r_[np.zeros(n),expected];upper=np.r_[np.ones(n),expected]
 result=milp(np.asarray(costs),integrality=np.ones(len(options)),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lower,upper),options={'time_limit':time_limit})
 if result.x is None:return {'state':'unresolved','message':result.message}
 selected=np.rint(result.x).astype(int);actual=matrix@selected
 if not np.array_equal(actual[n:],expected) or np.max(actual[:n])>1:return {'state':'unresolved','message':'non-feasible result'}
 for e in events:e['isFake']=True
 shifted=0
 for flag,(j,shift) in zip(selected,options):
  if not flag:continue
  e=events[j];e['isFake']=False
  if shift:
   e['timingAdjustment']=round(shift,5);e['originalTime']=e['time'];e['time']=round(max(0,e['time']+shift),5);e['sourceTime']=e['time'];shifted+=1
 for i,q in enumerate(intervals):q['estimatedJudgements']=int(actual[n+i]);q['difference']=0
 return {'state':'feasible-count-allocation','solverStatus':int(result.status),'frameAdjustedNotes':shifted,'note':'个别判定时刻在一帧视频采样不确定性内调整；像素轨迹不平移'}
