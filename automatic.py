"""Unattended video-only reconstruction with automatic speed calibration and repairs."""
import argparse,json,subprocess,sys,shutil,math,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
from export import export_js
from bundle import build_bundle
from shapes import normalize_holds
from timeline import refine,repair,settle
P=argparse.ArgumentParser();P.add_argument('video');P.add_argument('--output',required=True);P.add_argument('--title',default='Video reconstruction');P.add_argument('--resume',action='store_true');a=P.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True);here=Path(__file__).resolve().parent
statusfile=root/'progress.json'
fingerprint={'video':hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),'scanner':hashlib.sha256(b''.join((here/f).read_bytes() for f in ('reconstruct.py','combo.py','evidence.py'))).hexdigest()}
manifest=root/'scan-fingerprint.json';cache=a.resume and manifest.exists() and json.loads(manifest.read_text())==fingerprint
manifest.write_text(json.dumps(fingerprint))
def progress(phase):
 statusfile.write_text(json.dumps({'phase':phase},ensure_ascii=False));print(phase,flush=True)
def run(script,*args):
 if cache:
  values=list(map(str,args))
  dest=Path(values[values.index('--output')+1]) if '--output' in values else root
  needed={'reconstruct.py':['events.json','tracks.json','targets.json','report.json'],'combo.py':['combo.json'],'evidence.py':['effects.json','hit-evidence.json']}.get(script,[])
  if needed and all((dest/f).exists() for f in needed):return 'reused verified pixel scan cache'
 r=subprocess.run([sys.executable,str(here/script),*map(str,args)],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[-3000:])
 return r.stdout
progress('1/10：追踪视频轨迹，标定判定位置与像素流速')
base=root/'passes'/'168';run('reconstruct.py',a.video,'--output',base,'--title',a.title)
targets=np.array(json.loads((base/'targets.json').read_text())).reshape(-1,3)
judge=.82361
if len(targets):
 bins=np.round(targets[:,2]/2).astype(int);hist=np.bincount(bins,minlength=180);hist[:110]=0;mode=int(np.argmax(hist))*2
 if hist.max()>30 and mode>225:judge=mode/360
progress('2/10：自动扩大检测阈值，重新扫描遮挡与缺漏')
all_events=[]
for threshold in (168,150,188):
 dest=root/'passes'/str(threshold)
 if threshold!=168:run('reconstruct.py',a.video,'--output',dest,'--title',a.title,'--judge-y',judge,'--threshold',threshold,'--min-frames',3)
 events=json.loads((dest/'events.json').read_text())
 for e in events:e['pass']=threshold;e['isFake']=False;e['evidence']='visible-note-track'
 all_events.extend(events)
# Prefer long, consistent tracks; preserve distinct close notes when x differs.
chosen=[]
for e in sorted(all_events,key=lambda e:(e['frames'],e['confidence']=='medium'),reverse=True):
 if any(abs(e['time']-q['time'])<.058 and abs(e['x']-q['x'])<38 for q in chosen):continue
 chosen.append(e)
chosen.sort(key=lambda e:e['time'])
rejected_holds=normalize_holds(chosen,root/'passes')
progress('3/10：标定稳定流速与变速区间')
good=[e for e in chosen if e['frames']>=8 and 200<e['speed']<1600]
nominal=float(np.median([e['speed'] for e in good])) if good else 950.
segments=[];duration=json.loads((base/'report.json').read_text())['duration']
for start in np.arange(0,duration,8):
 values=np.array([e['speed'] for e in good if start<=e['time']<start+8]);med=float(np.median(values)) if len(values)>4 else nominal;mad=float(np.median(abs(values-med))) if len(values) else 0
 segments.append({'from':float(start),'to':float(min(start+8,duration)),'pixelsPerSecondAt1280':round(med,3),'samples':len(values),'medianAbsoluteDeviation':round(mad,3),'milplayFlowAtDefault':round(med/(120*1.66*720/1080),5)})
for e in chosen:
 seg=segments[min(len(segments)-1,int(e['time']//8))]
 # Unstable short tracks use calibrated segment speed. Preserve observed path positions.
 if e['frames']<6 or abs(e['speed']-seg['pixelsPerSecondAt1280'])>max(180,seg['medianAbsoluteDeviation']*4):
  old=e['speed'];e['speed']=seg['pixelsPerSecondAt1280']
  if e['duration']>0:e['duration']*=old/e['speed']
(root/'calibration.json').write_text(json.dumps({'judgeY':judge,'nominalPixelsPerSecondAt1280':nominal,'segments':segments,'note':'像素标定，不声称等同于官方游戏流速设置数值；导出自动补偿播放器 flow_speed。'},ensure_ascii=False,indent=2))
progress('4/10：从视频读取连击与击打特效，重建可见直线效果')
combo=None
if shutil.which('tesseract'):
 run('combo.py',a.video,'--output',root);combo=json.loads((root/'combo.json').read_text())
run('evidence.py',a.video,'--output',root)
effects=json.loads((root/'effects.json').read_text());hitframes=json.loads((root/'hit-evidence.json').read_text())
progress('5/10：融合视觉证据，自动补检与推断假音符')
# Preserve pixel hypotheses for the final ring/counter solver.
# Do not fabricate heads from purple note outlines or a final count budget.
repairs=[];fake_count=0;target_count=combo['finalCombo'] if combo else None
local_audit=[];allocation={'state':'awaiting-full-source-fit'}
count=sum(1+(e['duration']>0) for e in chosen)
# Independent decorative tracks: sustained motion, no plausible real-note counterpart.
track_grid=set()
for e in chosen:
 for t,x,y in e.get('path',[]):track_grid.add((round(t*15),round(x/40),round(y/40)))
def overlaps(s):
 matched=0
 for q in s[::2]:
  kt=round(float(q[0])*15);kx=round(float(q[1])*2/40);ky=round(float(q[2])*2/40)
  if any((kt+dt,kx+dx,ky+dy) in track_grid for dt in (-1,0,1) for dx in (-1,0,1) for dy in (-1,0,1)):matched+=1
 return matched/max(1,len(s[::2]))
tracks=json.loads((base/'tracks.json').read_text());used={e.get('track') for e in chosen if e.get('pass')==168}
for tr in tracks:
 if tr['id'] in used:continue
 s=np.array(tr['samples']);
 if len(s)<10 or np.linalg.norm(s[-1,1:3]-s[0,1:3])<55 or overlaps(s)>.5:continue
 t=float(s[-1,0]);x=float(s[-1,1]*2);y=float(s[-1,2]*2)
 if any(abs(t-e['time'])<.4 and abs(x-e['x'])<45 for e in chosen):continue
 # A fake hypothesis must end away from a detected judgement target.
 near=targets[np.abs(targets[:,0]-t)<.15]
 if len(near) and np.min(np.linalg.norm(near[:,1:3]-s[-1,1:3],axis=1))<25:continue
 dt=float(s[-1,0]-s[0,0]);speed=float(np.linalg.norm(s[-1,1:3]-s[0,1:3])*2/dt)
 if not 150<speed<1600:continue
 chosen.append({'time':t,'sourceTime':t,'x':x,'sourceX':x,'judgeY':y/720,'duration':0,'type':int(np.median(s[:,5])>=.5),'frames':len(s),'speed':speed,'confidence':'inferred','isFake':True,'evidence':'decorative-track-hypothesis','path':[[float(q[0]),float(q[1]*2),float(q[2]*2)] for q in s[::2]]});fake_count+=1
chosen.sort(key=lambda e:e['time'])
(root/'events.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2));(root/'repairs.json').write_text(json.dumps(repairs,ensure_ascii=False,indent=2))
progress('6/10：导出谱面、音频、曲绘与 milplay ZIP')
package={'file':'milplay.zip','state':'pending-full-source-fit'}
report={'automatic':True,'package':package,'videoDuration':duration,'visibleCandidates':len(chosen),'inferredFakeNotes':fake_count,'effectOnlyRepairs':len(repairs),'rejectedThinHoldCandidates':rejected_holds,'lineEffects':len(effects),'videoFinalCombo':target_count,'estimatedRealJudgements':count,'unresolvedCountDifference':target_count-count if target_count else None,'counterAllocation':allocation['state'],'localCounterIntervals':len(local_audit),'localCounterUnresolved':sum(abs(q['difference']) for q in local_audit),'nominalPixelsPerSecondAt1280':round(nominal,3),'judgeY':judge,'source':str(Path(a.video).resolve()),'limitations':['计数约束不等同于逐个音符准确匹配','特效补检的时间、类型与假音符标记属于推断','直线效果为像素拟合；完整故事板、隐藏音符与官方动画参数无法由视频唯一确定','OCR 遮挡可能导致计数错误；未自动证明与原谱完全一致']}
(root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));progress('6/10：基础扫描完成，继续复原移动判定几何')
shutil.copyfile(root/'events.json',root/'events-v1.json')
from upgrade import upgrade
print(json.dumps(upgrade(a.video,root,a.title,a.resume),ensure_ascii=False),flush=True)
