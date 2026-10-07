"""Playable Milize export with native judgement lines and measured note trajectories."""
import json,math
import numpy as np
from curve_fit import curves

# The detected core width is measured in source pixels, while the storyboard
# sprite's alpha profile occupies less than half of its nominal vertical span.
# Compensate for that texture response so the rendered streak matches the
# measured bright core instead of looking like a thin rail.
STREAK_TEXTURE_RESPONSE_GAIN=2.3

def simplify(points,tolerance=1):
 if len(points)<3:return points
 p=np.asarray(points,float);keep={0,len(p)-1};stack=[(0,len(p)-1)]
 while stack:
  a,b=stack.pop()
  if b-a<2:continue
  dt=p[b,0]-p[a,0]
  if dt<=0:continue
  frac=(p[a+1:b,0]-p[a,0])/dt;pred=p[a,1:]+frac[:,None]*(p[b,1:]-p[a,1:]);errors=np.max(abs(p[a+1:b,1:]-pred),axis=1);k=int(np.argmax(errors))+a+1
  if errors.max()>tolerance:keep.add(k);stack.extend([(a,k),(k,b)])
 return [[round(float(v),5) for v in p[i]] for i in sorted(keep)]

def extend_measured_approach(path,hit_time):
 """Extend a clearly observed fall toward the top, but avoid long guesses.

 A track that was only measured a frame or two before judgement does not
 support reconstructing its earlier screen path. Extrapolating it for most
 of a second can invent an isolated capsule before it joins an overlapping
 hold. Require a few continuous pre-hit samples before adding an entrance.
 """
 path=[list(q) for q in path]
 if len(path)<3:return path
 p=np.asarray(path,float);first,last=p[0],p[min(3,len(p)-1)];dt=last[0]-first[0]
 if dt<=.01:return path
 steps=np.diff(p[:min(5,len(p)),0]);steps=steps[steps>1e-4]
 sample_step=float(np.median(steps)) if len(steps) else 1/30
 minimum_lead=max(.08,min(.12,3*sample_step))
 if float(hit_time)-float(first[0])<minimum_lead:return path
 vx=(last[1]-first[1])/dt;vy=(last[2]-first[2])/dt
 if vy>300 and 0<first[2]<680:
  extra=min(.75,(first[2]+28)/vy,float(first[0]))
  if extra>0:path.insert(0,[float(first[0]-extra),float(first[1]-vx*extra),float(first[2]-vy*extra)])
 return path

def moving_hold_anchor(event,anchor):
 """Use the measured post-hit head motion when a hold's fitted anchor is static.

 Holds on moving judgement lines keep their head on the moving line after the
 hit. Some ring fits keep moving after they stop describing the physical head,
 while others are merely noisy for one frame. Prefer the ring fit unless a
 high-resolution capsule-cap track disagrees with it repeatedly. Preserve the
 selected judgement point and use the measured head only after contact.
 """
 t=float(event['time']);end=t+float(event.get('duration',0) or 0)
 if event.get('isFake') or end<=t:return anchor
 path=event.get('bodyPath') or event.get('path') or []
 observed=[]
 for q in path:
  if len(q)<3:continue
  tm,x,y=map(float,q[:3])
  # The last decoded source frame can land up to one frame after the inferred
  # release time. Keep that measurement, but clamp it to the playable endpoint.
  if t<tm<=end+.035:observed.append([min(tm,end),x,y])
 observed.sort(key=lambda q:q[0])
 if observed:
  deduped=[]
  for q in observed:
   if deduped and abs(q[0]-deduped[-1][0])<1e-5:deduped[-1]=q
   else:deduped.append(q)
  observed=deduped
 if len(observed)<2:return anchor
 existing=np.asarray(anchor,float).copy() if len(anchor)>1 else np.empty((0,3))
 anchor_repaired=False
 if (event.get('bodyEvidence')=='full-resolution-blue-capsule-cap-pair'
     and len(existing)>=2 and len(observed)>=3):
  existing=existing[np.argsort(existing[:,0],kind='stable')]
  _,indices=np.unique(existing[:,0],return_index=True)
  existing=existing[np.sort(indices)]
  # Ring fits occasionally contain one displaced anchor where the continuous
  # capsule track and both neighboring anchors agree on a smooth path. Repair
  # only that isolated sample; preserve intentional bends and measured motion.
  if len(existing)>=3:
   for i in range(1,len(existing)-1):
    left,right=existing[i-1],existing[i+1]
    if right[0]-left[0]>.12 or not left[0]<existing[i,0]<right[0]:continue
    fraction=(existing[i,0]-left[0])/(right[0]-left[0])
    predicted=left[1:]+fraction*(right[1:]-left[1:])
    nearest=min(observed,key=lambda q:abs(q[0]-existing[i,0]))
    if abs(nearest[0]-existing[i,0])>.02:continue
    displacement=float(np.linalg.norm(existing[i,1:]-predicted))
    measured_error=float(np.linalg.norm(np.asarray(nearest[1:])-predicted))
    if displacement>6. and measured_error<4.:
     existing[i,1:]=nearest[1:]
     anchor_repaired=True
  errors=[]
  for tm,x,y in observed:
   ax=float(np.interp(tm,existing[:,0],existing[:,1]))
   ay=float(np.interp(tm,existing[:,0],existing[:,2]))
   errors.append(float(np.hypot(ax-x,ay-y)))
  # A single tracking jump can come from a crossing note or a glow fragment.
  # For an uninterrupted high-resolution cap track, three consecutive misses
  # above 6 px are also meaningful: even a small angle/offset compounds along
  # a long hold. Keep the older two-sample guard for larger displacements.
  consecutive=0;longest=0;previous_time=None
  for (tm,_,_),error in zip(observed,errors):
   if error>6. and (previous_time is None or tm-previous_time<=.05):consecutive+=1
   elif error>6.:consecutive=1
   else:consecutive=0
   longest=max(longest,consecutive);previous_time=tm
  if longest>=3 or sum(error>12. for error in errors)>=2:
   result=[[t,float(event['x']),float(event.get('judgeY',.822222))*720],*observed]
   if result[-1][0]<end:result.append([end,result[-1][1],result[-1][2]])
   return result
 movement=float(np.hypot(observed[-1][1]-observed[0][1],observed[-1][2]-observed[0][2]))
 if movement<8:return existing.tolist() if anchor_repaired else anchor
 existing_motion=(float(np.hypot(np.ptp(existing[:,1]),np.ptp(existing[:,2])))
                  if len(existing)>1 else 0.)
 if existing_motion>=max(6.,movement*.4):return existing.tolist() if anchor_repaired else anchor
 x=float(event['x']);y=float(event.get('judgeY',.822222))*720
 result=[[t,x,y],*observed]
 if result[-1][0]<end:result.append([end,result[-1][1],result[-1][2]])
 return result

def fallback_approach_lead(event,path):
 """Limit an inferred hit with no measured sprite path to a near-contact cue.

 Counter/ring repairs can establish that a judgement occurred without
 establishing where its note travelled beforehand.  The ordinary default
 approach is useful when there is a real source path, but drawing six tenths
 of a second of guessed motion can create a slow phantom note.  Keep a very
 short lead for these unsupported approaches so the chart retains a playable
 contact cue without claiming an unobserved trajectory.
 """
 if not path and not event.get('isFake') and event.get('confidence')=='inferred':
  return .02
 return .6

def filter_disconnected_rail_fragments(effects):
 """Discard short rails whose off-screen fallback jumps onto a target."""
 kept=[]
 for effect in effects or []:
  samples=effect.get('samples') or []
  discard=False
  if effect.get('kind')=='rail' and 2<=len(samples)<=3:
   for left,right in zip(samples,samples[1:]):
    dt=float(right.get('time',0))-float(left.get('time',0))
    if dt<=0 or dt>.05:continue
    if not left.get('nativeAnchor') or not right.get('nativeAnchor'):continue
    if bool(left.get('anchored'))==bool(right.get('anchored')):continue
    distance=math.hypot(float(right['x'])-float(left['x']),float(right['y'])-float(left['y']))
    if distance>1800:
     discard=True
     break
  if not discard:kept.append(effect)
 return kept

def export_js(events,title='Video reconstruction',judge_y=.822222,effects=None,duration=141.04):
 from axis_regularize_v6 import regularize
 effects,_=regularize(effects)
 effects=filter_disconnected_rail_fragments(effects)
 out=['// Video-only reconstruction: native lines and source-observed capsule endpoints.','const m=MilizeBeatmap;',f'm.withProperty("Title",{json.dumps(title,ensure_ascii=False)}).withProperty("Difficulty","Cloudburst").withProperty("Beatmapper","Video reconstruction").withProperty("AudioFile","audio.m4a").withProperty("IllustrationFile","cover.jpg");','const b=m.timing(0,60,4),fc=1.66/(Number(m.env("user.flow_speed"))||1.66);', 'const vt=t=>Math.max(0,Math.round((t-.00002)*1e6)/1e6);function a(o,d,k,p){if(!p.length)return;m.animation(b,vt(p[0][0]),vt(p[0][0]),k,p[0][1],p[0][1],d,o,0,0,false,"");for(let i=1;i<p.length;i++){let u=p[i-1],v=p[i];m.animation(b,vt(u[0]),vt(v[0]),k,u[1],v[1],d,o,0,0,false,"");}}', 'function xy(o,d,p){a(o,d,0,p.map(v=>[v[0],v[1]]));a(o,d,1,p.map(v=>[v[0],v[2]]));}', 'function vis(o,d,start,end,alpha=1){a(o,d,2,[[0,0],[start,0],[start+.00001,alpha],[end,alpha],[end+.00001,0]]);}', '{let s=m.storyboardObject(0,"storyboard/background.png",1);a(s,2,10,[[0,1.5]]);a(s,2,11,[[0,1.5]]);a(s,2,2,[[0,1],['+str(max(0,duration-1))+',1]]);}']
 def data(p):return json.dumps(p,separators=(',',':'),ensure_ascii=False)
 out.append('function c(o,d,k,p){if(p.length===1){let q=p[0];m.animation(b,vt(q[0]),vt(q[0]),k,q[1],q[1],d,o,0,0,false,"");}for(let i=1;i<p.length;i++){let u=p[i-1],v=p[i];m.animation(b,vt(u[0]),vt(v[0]),k,u[1],v[1],d,o,0,0,false,"");}}')
 def emit(o,d,k,p,tol=1.):
  segments=curves(p,tol)
  if not segments:return ''
  compact=[[segments[0][0],round(segments[0][2],3)]]+[[s[1],round(s[3],3)] for s in segments if s[1]>s[0]]
  return f'c({o},{d},{k},'+data(compact)+');'
 def emit_xy(o,d,p,tol=1.,split=None):
  groups=[p] if split is None else [[q for q in p if q[0]<=split],[q for q in p if q[0]>=split]]
  return ''.join(emit(o,d,k,[[q[0],q[k+1]] for q in group],tol) for group in groups if group for k in (0,1))
 carriers=[];out.append('const ln=[];')
 for e in sorted(events,key=lambda e:e['time']):
  # Rejected hit-only hypotheses have no observed sprite. Drawing a default
  # approach for them created phantom white taps amongst real hollow chains.
  if e.get('isFake') and not e.get('path') and e.get('evidence')=='video-counter-and-fresh-source-hit-ring':continue
  t=max(0,float(e['time']));dur=max(0,float(e.get('duration',0)));end=t+dur;note_end=end;visual_end=end;fake=bool(e.get('isFake'));x=float(e['x']);y=float(e.get('judgeY',judge_y))*720;rot=float(e.get('rotation',90));path=[list(q) for q in e.get('path',[])]
  # Source segmentation can begin after a note has already crossed a large
  # part of the playfield.  For every real, clearly downward-moving track,
  # reconstruct the missing approach so the note enters from above instead
  # of popping into view halfway down the screen.  Decorative/fake tracks are
  # intentionally excluded because their paths are often stationary effects.
  if not fake:path=extend_measured_approach(path,t)
  if not all(math.isfinite(v) for v in (t,dur,x,y,rot)):raise ValueError('音符包含无效数字')
  if int(e['type']) not in (0,1,2):raise ValueError('音符类型无效')
  # Decorative notes are still rendered notes.  Their observed trajectory is
  # the visual lifetime; placing a zero-duration fake at path[-1] makes it
  # flash for one frame at the end of the track.
  if fake and path:
   observed_start=float(path[0][0]);observed_end=float(path[-1][0])
   visual_end=max(float(end),observed_end+1/30)
   # Milplay stops drawing a non-hold note 0.16s after its nominal hit time.
   # A decorative fake drag can remain visible much longer, so anchor its
   # non-scoring hit time just after the measured visual lifetime. Its explicit
   # path and opacity tracks still control where and when it appears.
   t=visual_end if dur<=0 else max(float(t),observed_start)
   note_end=t+dur
   # A non-scoring Hold may remain visibly on screen after the inferred event
   # duration. Keeping its native endSec shorter than that observed lifetime
   # makes milplay calculate a short tail, hiding the upper half of the body.
   # Fake Holds have no combo or release judgement, so use their measured visual
   # endpoint as the playback endpoint as well.
   if dur>0:note_end=visual_end
  rotations=np.asarray(e.get('holdRotationPath',[[t,rot]]),float)
  def rotation_at(tm):return float(np.interp(tm,rotations[:,0],rotations[:,1]))
  rot=rotation_at(t) if dur>0 else (90 if not fake else round(rot/2)*2)
  base_y=round(y) if not fake else 360
  fallback_lead=fallback_approach_lead(e,path)
  start=min(t,min(float(q[0]) for q in path)) if path else max(0,t-fallback_lead)
  slot=next((q for q in carriers if q[1]+.001<start and q[2]==fake),None)
  if slot is None:
   idx=len(carriers);slot=[idx,visual_end+.55,fake];carriers.append(slot)
   out.append(f'ln[{idx}]=m.line();a(ln[{idx}],0,0,[[0,0]]);a(ln[{idx}],0,8,[[0,0]]);a(ln[{idx}],0,9,[[0,0]]);a(ln[{idx}],0,23,[[0,1000000]]);')
  else:slot[1]=visual_end+.55
  anchor=moving_hold_anchor(e,e.get('anchorPath',[[t,x,y],[end,x,y]]))
  anchor=np.asarray(anchor,float)
  if not fake:
   # If a counter fit moves a judgement by a few frames, make the animated
   # origin pass through the selected hit point at that time.
   anchor=anchor[np.abs(anchor[:,0]-t)>1e-5]
   anchor=np.vstack((anchor,np.asarray([[t,x,y]],float)))
   anchor=anchor[np.argsort(anchor[:,0],kind='stable')]
  if not fake:
   carrier=[[start,(anchor[0,1]-640)*1.5,(360-anchor[0,2])*1.5]]+[[float(q[0]),(float(q[1])-640)*1.5,(360-float(q[2]))*1.5] for q in anchor]
   out.append(emit_xy(f'ln[{slot[0]}]',0,carrier,.8))
  else:
   out.append(f'a(ln[{slot[0]}],0,0,[[{start:.5f},0]]);a(ln[{slot[0]}],0,1,[[{start:.5f},{(360-base_y)*1.5:.5f}]]);')
  if dur>0 and e.get('holdRotationPath'):
   out.append(emit(f'ln[{slot[0]}]',0,4,[[start,rotation_at(start)]]+[[float(q[0]),float(q[1])] for q in rotations if start<float(q[0])<visual_end]+[[visual_end,rotation_at(visual_end)]],.45))
  else:out.append(f'a(ln[{slot[0]}],0,4,[[{start:.5f},{rot}]]);')
  out.append(f'{{let n=m.note(ln[{slot[0]}],b,{t:.5f},{note_end:.5f},{int(e["type"])},{str(fake).lower()},{str(bool(e.get("isAlwaysPerfect",False))).lower()});')
  default_flow=float(e.get('speed',960))/(120*1.66*720/1080)
  theta=math.radians(90-rot);ca,sa=math.cos(theta),math.sin(theta);shift=0;xs=0.;points=[]
  def local(tm,px,py):
   theta=math.radians(90-rotation_at(tm) if dur>0 else 90-rot);ca,sa=math.cos(theta),math.sin(theta)
   ax=640 if fake else float(np.interp(tm,anchor[:,0],anchor[:,1]))
   ay=base_y if fake else float(np.interp(tm,anchor[:,0],anchor[:,2]))
   dx=(float(px)-ax)*1.5;dy=(ay-float(py))*1.5
   return [max(0,float(tm)),dx*ca-dy*sa,dx*sa+dy*ca]
  # A hold head is pinned to the judgement line after it is hit.  The engine
  # owns the tail contraction through floorEnd, so post-hit pixel samples must
  # not keep animating the whole hold body and hide its upper section.
  cutoff=visual_end if fake or dur<=0 else t
  for tm,px,py in path:
   tm=float(tm)+shift
   if tm>cutoff:continue
   points.append(local(tm,px,py))
  if not fake and dur<=0 and len(path)>2 and path[-1][0]<t+.12:
   tail=path[-min(6,len(path)):];good=[p for p in tail if p[0]<t+.01]
   if len(good)>1:
    dt=good[-1][0]-good[0][0];vx=(good[-1][1]-good[0][1])/max(.01,dt);vy=(good[-1][2]-good[0][2])/max(.01,dt);points.append(local(t+.16,x+vx*.16,y+vy*.16))
  if not fake:
   points=[p for p in points if abs(p[0]-t)>.0001]
   # The judgement point is an explicit anchor sample, so the local note
   # origin is zero at hit time and cannot drift away from the animated line.
   points.append([t,0.,0.]);points.sort(key=lambda p:p[0])
  if not path:
   centre=local(t,x,y);px,py=centre[1:];speed=float(e.get('speed',960))*1.5;lead=max(0,t-start);points=[[start,px,py+speed*lead],[t,px,py],[t+.16,px,py-speed*.16]]
  if len(points)>=1:
   out.append(emit_xy('n',1,points,1.5,None if fake else t))
   start=points[0][0];finish=visual_end+.16 if not fake else visual_end
   # Zero size outside the observed lifetime lets milplay reject the note
   # before evaluating its position/FLOW tracks. This is a native animation,
   # with no per-frame expression or runtime patch.
   scale=float(e.get('noteScale',1.));edge=max(0,start-.00002)
   widths=e.get('holdWidthPath',[]) if dur>0 else []
   width_points=[[q[0],max(.2,q[1]/(2000*.0223*(335/185)*.55))] for q in widths if start<=q[0]<=finish]
   if width_points:scale=width_points[0][1]
   envelope=([[0,scale]] if edge<=0 else [[0,0],[edge,0],[edge+.00001,scale]])+simplify(width_points,.04)+[[finish,width_points[-1][1] if width_points else scale],[finish+.00001,0]]
   out.append('a(n,1,3,'+data(envelope)+');')
  if e.get('opacityPath'):
   pp=[[q[0],q[1]] for q in e['opacityPath'] if start<=q[0]<=finish]
   if pp:out.append('a(n,1,2,'+data([[0,0],[max(0,start-.00002),0],[start,pp[0][1]]]+simplify(pp,.07)+[[finish,pp[-1][1]],[finish+.00001,0]])+');')
  orientation=e.get('orientationPath',[])
  if orientation:
   pp=[[max(0,float(q[0])+shift),(-float(q[1])-rotation_at(float(q[0])))] for q in orientation if float(q[0])+shift<cutoff]
   if pp:out.append('a(n,1,4,'+data(simplify(pp,4))+');')
  # Fit the visible held body rather than letting an estimated constant flow stretch it.
  flow_points=[]
  if dur>0 and e.get('tailPath') and path:
   heads=np.asarray(path,float);pp=[]
   for tm,tx,ty in e['tailPath']:
    tm=float(tm)+shift
    if tm>=visual_end-.015:continue
    hx=float(np.interp(tm,heads[:,0],heads[:,1]));hy=float(np.interp(tm,heads[:,0],heads[:,2]))
    if not fake and tm>=t:
     hx=float(np.interp(tm,anchor[:,0],anchor[:,1]));hy=float(np.interp(tm,anchor[:,0],anchor[:,2]))
    theta=math.radians(90-rotation_at(tm));ca,sa=math.cos(theta),math.sin(theta)
    dx=(float(tx)-hx)*1.5;dy=(hy-float(ty))*1.5;length=max(0,dx*sa+dy*ca);flow=length/(max(.015,visual_end-max(t,tm))*120*1.66)
    pp.append([max(0,tm),flow])
   if pp:
    flow_points=[[0,pp[0][1]]]+simplify(pp,.05)
  if not flow_points:flow_points=[[0,default_flow]]
  out.append('a(n,1,5,'+data(flow_points)+'.map(p=>[p[0],p[1]*fc]));')
  out.append('}')
 # Reuse native line bearers once their observed visibility window has ended.
 # Detached Hough fragments should not allocate thousands of permanently
 # iterated line objects, unlike the small reusable line sets in normal charts.
 pools={'target':[],'rail':[]};out.append('const fx=[],sbx=[];');streak_index=0
 for effect in sorted(effects or [],key=lambda e:e['samples'][0]['time'] if e.get('samples') else 0):
  ss=effect.get('samples',[])
  if not ss:continue
  start,end=max(0,float(ss[0]['time'])-.00001),float(ss[-1]['time'])+1/30-.00002
  kind=effect.get('kind','rail')
  if kind=='streak':
   obj=f'sbx[{streak_index}]';streak_index+=1
   out.append(f'{obj}=m.storyboardObject(0,"storyboard/light-streak.png",2);')
   positions=[[float(q['time']),(float(q['x'])/1280-.5)*1920,(.5-float(q['y'])/720)*1080] for q in ss]
   # Detector angles use image coordinates (positive y points down) and describe
   # an unoriented axis modulo 180 degrees. Milplay's optimized storyboard path
   # negates authored Rotation while drawing, so convert the sign here and unwrap
   # the axis before interpolation to prevent a 180-degree flip at the seam.
   axis=[]
   for q in ss:
    angle=float(q['rotation'])
    if axis:
     while angle-axis[-1]>90:angle-=180
     while angle-axis[-1]<-90:angle+=180
    axis.append(angle)
   rotations=[[float(q['time']),-angle] for q,angle in zip(ss,axis)]
   widths=[[float(q['time']),float(q['length'])/(1280/1920*1024)] for q in ss]
   heights=[[float(q['time']),float(q['width'])*8/(max(1,float(q['length']))*.83)*STREAK_TEXTURE_RESPONSE_GAIN] for q in ss]
   alpha=[[max(0,start-.00002),0],[float(ss[0]['time']),float(ss[0].get('alpha',.9))]]+[[float(q['time']),float(q.get('alpha',.9))] for q in ss[1:]]+[[end,float(ss[-1].get('alpha',.9))],[end+.00001,0]]
   out.append(emit_xy(obj,2,positions,.25))
   out.append(emit(obj,2,4,rotations,.25)+emit(obj,2,10,widths,.015)+emit(obj,2,11,heights,.01))
   out.append(emit(obj,2,2,alpha,.03))
   continue
  pool=pools[kind];slot=next((q for q in pool if q[1]+.0001<start),None)
  if slot is None:
   idx=sum(len(q) for q in pools.values());slot=[idx,end];pool.append(slot)
   out.append(f'fx[{idx}]=m.line();a(fx[{idx}],0,2,[[0,0]]);a(fx[{idx}],0,8,[[0,{0 if kind=="target" else 1}]]);a(fx[{idx}],0,9,[[0,{1 if kind=="target" else 0}]]);')
  else:slot[1]=end
  out.append(f'{{let l=fx[{slot[0]}];')
  if effect.get('kind')=='target':
   pos=np.asarray([[q['x'],q['y']] for q in ss],float)
   for k in (0,1):
    if np.ptp(pos[:,k])<=4:pos[:,k]=np.median(pos[:,k])
   p=[[q['time'],(pos[i,0]/1280-.5)*1920,(.5-pos[i,1]/720)*1080] for i,q in enumerate(ss)]
   p.insert(0,[start,*p[0][1:]])
   out.append(emit_xy('l',0,p,3.5))
   out.append(emit('l',0,3,[[start,ss[0]['radius']/20.45366]]+[[q['time'],q['radius']/20.45366] for q in ss],.035))
   alpha=[[max(0,start-.00002),0],[start,ss[0]['alpha']]]+[[q['time'],q['alpha']] for q in ss[1:]]+[[end,ss[-1]['alpha']],[end+.00001,0]]
   # Visibility edges are explicit; avoid simplification changing their timings.
   out.append(emit('l',0,2,alpha,.09)+'}')
  else:
   # Rails are gameplay line bodies rather than storyboard textures.  Their
   # source segments are observations of an effectively unbounded judgement
   # line; anchoring the body at the observed start point keeps it continuous
   # when the chart is rendered at another aspect ratio.
   def rail_points(key):
    return [[float(q['time']),float(q[key])] for q in ss]
   def endpoint(q):
    if q.get('nativeAnchor'):
     # line_reconstruct_v6.sample() stores native anchors directly in the
     # milplay UI coordinate system.  Do not treat them as source pixels and
     # apply the pixel-to-UI transform a second time here; that pushed dense
     # horizontal/vertical line arrays thousands of units off screen.
     return float(q['x']),float(q['y']),float(q['rotation'])
    x1,y1,x2,y2=[float(v) for v in q.get('ends',[0,0,0,0])]
    length=max(1e-6,math.hypot(x2-x1,y2-y1))
    # Use the observed segment direction, then extend its start well beyond
    # the camera.  A native gameplay line is one-sided (its body starts at the
    # head and runs forward), so using the finite Hough segment endpoint made
    # the line disappear above the first observed pixel and failed on other
    # aspect ratios.  The extra span is deliberately larger than either
    # landscape or portrait viewport.
    dx,dy=(x2-x1)/length,(y2-y1)/length
    rot=float(q.get('rotation',90))
    rd=(math.cos(math.radians(rot)),math.sin(math.radians(rot)))
    if dx*rd[0]+dy*rd[1]<0: dx,dy=-dx,-dy
    extend=2200.0
    sx=x1-dx*extend; sy=y1-dy*extend
    # The native line prefab starts 11.54 UI units after its centre.
    connect=11.54*1.6
    sx-=dx*connect; sy-=dy*connect
    return (sx/1280-.5)*1920,(.5-sy/720)*1080,-rot
   posx=[];posy=[];rot=[];size=[];alpha=[[max(0,start-.00002),0],[start,float(ss[0].get('alpha',.75))]]
   for q in ss:
    px,py,rr=endpoint(q);posx.append([float(q['time']),px]);posy.append([float(q['time']),py]);rot.append([float(q['time']),rr]);
    ui=1.6
    # `width` in rails-v3 is a segment-length proxy from the pixel detector,
    # not the native line thickness.  Keep a stable gameplay width so the
    # line remains visible at any viewport size; overlapping observations then
    # naturally reproduce the brighter/broader sections in the video.
    size.append([float(q['time']),float(q.get('size',1.0))])
    alpha.append([float(q['time']),float(q.get('alpha',.75))])
   alpha += [[end,float(ss[-1].get('alpha',.75))],[end+.00001,0]]
   # Pooled geometry must reset before alpha becomes visible. Rounded video
   # frame times otherwise show the preceding object's position for 10 µs.
   for pp in (posx,posy,rot,size):pp.insert(0,[start,pp[0][1]])
   out.append(emit('l',0,0,posx,.3)+emit('l',0,1,posy,.3))
   # A sub-degree error at an off-screen origin displaces the visible rail
   # by many pixels. Keep rotation tighter than ordinary note orientation.
   out.append(emit('l',0,4,rot,.025)+emit('l',0,3,size,.08))
   out.append(emit('l',0,2,alpha,.05)+'}')
 return '\n'.join(out)+'\n'
