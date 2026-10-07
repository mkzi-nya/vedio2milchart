"""Unattended second pass: dynamic judgement geometry, full-screen notes and playable ZIP."""
import bisect,copy,json,hashlib,shutil,subprocess
import cv2
from pathlib import Path
from geometry import scan as geometry_scan
from notes_v2 import scan as note_scan
from enrich_v2 import enrich
from decorative import scan as decorative_scan,merge as merge_decorations
from hold_fit import fit_holds
from hold_refine_v4 import refine as refine_capsules,consolidate
from judgement_fit_v4 import recover_approaches,fit as fit_judgements
from effect_anchor_v5 import scan as scan_hit_rings,fit as fit_effect_anchors,supplement_hits
from anchors import dense_combo
from timeline import refine,repair,settle,settle_with_frame_uncertainty
from export_v2 import export_js,filter_disconnected_rail_fragments
from bundle import build_bundle
from offscreen_holds_v5 import extend as extend_offscreen_holds
from local_reconcile import align_weak_heads as align_counter_timing
from local_reconcile import reconcile as reconcile_local_counts
from counter_guard import guard_fit as guard_counter_fit, pin_drifted_contacts
HERE=Path(__file__).resolve().parent

def prune_visual_noise(events):
 """Drop transient detector fragments before assigning chart note indices.

 A playable note has either a sustained path or independent timing evidence.
 One-frame hollow fragments are usually particles/outline remnants; exporting
 them as fake notes creates visible flashes and consumes a line carrier.
 Keep candidates backed by a fresh hit ring, a counter repair, or a measured
 approach even when their visible lifetime is short (this preserves lightning
 and occluded notes).
 """
 kept=[];removed=[]
 supported_evidence={
  'video-counter-and-fresh-source-hit-ring',
  'video-counter-track-repair',
  'video-counter-boundary-repair',
  'hit-effect-and-local-counter',
 }
 for e in events:
  if not e.get('isFake'):
   kept.append(e);continue
  evidence=e.get('evidence','');frames=int(e.get('frames',0));path=e.get('path',[]) or []
  ring=float(e.get('ringContactSupport',0) or 0)
  head=float(e.get('ringHeadSupport',0) or 0)
  supported=(ring>=.55 or head>=.45 or e.get('counterRequired') or
             e.get('approachEvidence') or evidence in supported_evidence)
  span=0.
  if len(path)>=2:
   try:span=max(float(path[-1][0])-float(path[0][0]),0.)
   except (TypeError,ValueError):span=0.
  transient=(evidence=='source-hollow-single-frame-sprite' and frames<3)
  fragmented=(frames<4 and len(path)<4 and not supported)
  short_decoration=(evidence in {'full-screen-decorative-track',
                                 'independent-faint-source-approach',
                                 'faint-hollow-moving-decoration',
                                 'occluded-independent-pass'} and
                    (frames<7 or span<.12) and not supported)
  if transient or fragmented or short_decoration:
   removed.append(e);continue
  kept.append(e)
 return kept,removed

def fingerprint(video,files):
 return {'video':hashlib.sha256(Path(video).read_bytes()).hexdigest(),'code':hashlib.sha256(b''.join((HERE/f).read_bytes() for f in files)).hexdigest()}

def gameplay_session(root):
 result=subprocess.run(['node',str(HERE/'verify_session.cjs'),str(root),'--report-only'],
                       capture_output=True,text=True)
 if result.returncode:
  raise RuntimeError('playability session review failed: '+result.stderr[-1200:])
 first=next((line for line in result.stdout.splitlines() if line.startswith('{')),None)
 if not first:raise RuntimeError('playability session review returned no JSON result')
 return json.loads(first)

def upgrade(video,root,title='Video reconstruction',resume=False):
 root=Path(root)
 def progress(text):(root/'progress.json').write_text(json.dumps({'phase':text},ensure_ascii=False));print(text,flush=True)
 if not (root/'events-v1.json').exists():shutil.copyfile(root/'events.json',root/'events-v1.json')
 for name,files,outputs,fn in [('geometry',['geometry.py'],['targets-v2.json','rails-v2.json'],geometry_scan),('notes',['geometry.py','notes_v2.py'],['notes-v2.json','tracks-v2.json'],note_scan)]:
  fp=fingerprint(video,files);cache=root/(name+'-cache.json');valid=resume and cache.exists() and json.loads(cache.read_text())==fp and all((root/f).exists() for f in outputs)
  progress('7/10：完整追踪移动判定圈、圆阵与有限轨道' if name=='geometry' else '8/10：重新识别全屏音符、假音符轨迹与长条头尾')
  if not valid:fn(video,root);cache.write_text(json.dumps(fp))
 fp=fingerprint(video,['decorative.py']);cache=root/'decorative-cache.json'
 faint_fp=fingerprint(video,['geometry.py','notes_v2.py']);faint_cache=root/'faint-notes-cache.json'
 if not(resume and faint_cache.exists() and json.loads(faint_cache.read_text())==faint_fp and (root/'notes-faint-v5.json').exists()):
  note_scan(video,root,faint=True);faint_cache.write_text(json.dumps(faint_fp))
 if not (resume and cache.exists() and json.loads(cache.read_text())==fp and (root/'decorative-v2.json').exists()):decorative_scan(video,root);cache.write_text(json.dumps(fp))
 progress('9/10：拟合长条接触、方向、AP 光晕，并校验分段连击')
 events,visuals,summary=enrich(video,root)
 before=len(visuals);visuals=filter_disconnected_rail_fragments(visuals)
 summary['disconnectedRailFragmentsRemoved']=before-len(visuals)
 summary['holdContactsRefitted']=fit_holds(events)
 summary['capsuleRefinement']=refine_capsules(video,root,events)
 summary['capsuleConsolidation']=consolidate(events)
 targets=json.loads((root/'targets-v2.json').read_text())
 summary['occludedApproachesRecovered']=recover_approaches(events,targets,json.loads((root/'hit-evidence.json').read_text()))
 combo=dense_combo(json.loads((root/'combo.json').read_text()));intervals=[]
 # A purple note outline is not a hit. The old broad-colour repair created
 # synthetic real notes from moving outlines; only ring-backed candidates
 # may participate in the final source-counter allocation.
 repairs=[]
 targets=json.loads((root/'targets-v2.json').read_text())
 for e in repairs:
  matches=[q for tr in targets for q in tr['samples'] if q['radius']<28 and abs(q['time']-e['time'])<.09 and abs(q['x']-e['x'])<90 and abs(q['y']-e['judgeY']*720)<90]
  if matches:
   q=min(matches,key=lambda q:abs(q['x']-e['x'])+abs(q['y']-e['judgeY']*720)+abs(q['time']-e['time'])*100);e['x']=e['sourceX']=q['x'];e['judgeY']=q['y']/720;e['evidence']='counter-and-hit-effect-at-visible-target'
 allocation={'state':'awaiting-source-ring-and-frame-counter-fit'}
 summary['faintDecorationSupplements']=merge_decorations(events,json.loads((root/'decorative-v2.json').read_text()))
 from hollow_notes_v6 import merge as merge_hollow
 summary['hollowNoteReview']=merge_hollow(video,root,events,targets)
 from hold_motion_v6 import refine as refine_hold_motion,apply_anchors as apply_capsule_anchors,merge_repairs
 summary['capsuleMotionReview']=refine_hold_motion(video,root,events)
 summary['sourceEffectAnchors']=fit_effect_anchors(events,targets,scan_hit_rings(video,root))
 summary['capsuleMotionReview'].update(merge_repairs(events))
 (root/'events-observed-v5.json').write_text(json.dumps(events,ensure_ascii=False))
 # A fresh output directory has no legacy probe file.  Generate the dense
 # source-counter observations here so the MILP is fully automatic; relying
 # on a previous review directory silently skipped combo fitting on first run.
 intervals=[];counter_seed=None;counter_observations=None
 if shutil.which('tesseract') and not (root/'counter-probes.json').exists():
  from counter_probe import scan as scan_counter_probes
  progress('9/10：逐帧读取视频连击，建立局部计数约束')
  scan_counter_probes(video,root)
 if (root/'counter-probes.json').exists():
  from counter_frames_v4 import scan as scan_counter
  cache=root/'counter-frames-v4.json';signature=hashlib.sha256((HERE/'counter_frames_v4.py').read_bytes()+(root/'counter-probes.json').read_bytes()).hexdigest()
  manifest=root/'counter-frames-v4-cache.json'
  if not (cache.exists() and manifest.exists() and json.loads(manifest.read_text()).get('signature')==signature):
   scan_counter(video,root);manifest.write_text(json.dumps({'signature':signature}))
  observation=json.loads(cache.read_text());counter_observations=observation['samples']
  cap=cv2.VideoCapture(str(video));source_fps=cap.get(cv2.CAP_PROP_FPS) or 30.;cap.release()
  counter_seed=copy.deepcopy(events)
  allocation=fit_judgements(events,counter_observations,combo['finalCombo'])
  # A later refit can displace a strong contact that was stable in the first
  # fit. Recheck after each pass so every newly displaced ring+head contact is
  # pinned before the next local allocation.
  pinned_contacts=0;contact_protection_passes=0
  for contact_protection_passes in range(1,7):
   newly_pinned=pin_drifted_contacts(events)
   if not newly_pinned:
    contact_protection_passes-=1
    break
   pinned_contacts+=len(newly_pinned)
   allocation=fit_judgements(events,counter_observations,combo['finalCombo'])
  summary['directContactProtectionPasses']=contact_protection_passes
  summary['directContactsKeptAtSourceTime']=pinned_contacts
  repairs=[]
  for iteration in range(2):
   additions=supplement_hits(events,allocation,scan_hit_rings(video,root))
   if not additions:break
   repairs.extend(additions);allocation=fit_judgements(events,counter_observations,combo['finalCombo'])
  summary['inferredEffectDelaySeconds']=next((e['sourceEffectDelay'] for e in events if 'sourceEffectDelay' in e),None)
  from repair_approach_v5 import attach as attach_repair_approaches
  summary['repairApproaches']=attach_repair_approaches(events,root)
  # Re-sample moving hold anchors at the fitted head time; do not retime the
  # source observations merely to satisfy a counter constraint.
  summary['sourceEffectAnchors']=fit_effect_anchors(events,targets,scan_hit_rings(video,root),retime=False)
  apply_capsule_anchors(events)
  summary['offscreenMovingHolds']=extend_offscreen_holds(video,events)
  (root/'judgement-fit-v4.json').write_text(json.dumps(allocation,ensure_ascii=False,indent=2))
  summary['videoFrameFit']={k:v for k,v in allocation.items() if k!='cumulative'}
  if allocation['state']=='fitted-video-frame-counter':
   previous=(0.,0,0)
   intervals=[]
   for q in allocation['cumulative']:
    start,prev,prev_actual=previous;expected=q['videoCombo']-prev
    actual=q['reconstructedCombo']-prev_actual
    intervals.append({'from':start,'to':q['time'],'videoJudgements':expected,'estimatedJudgements':actual,'difference':expected-actual});previous=(q['time'],q['videoCombo'],q['reconstructedCombo'])
  from contact_spacing_v1 import adjust as separate_close_contacts
  summary['contactSpacing']=separate_close_contacts(events,source_fps)
 if counter_seed is not None and counter_observations is not None:
  events,summary['counterFitGuard']=guard_counter_fit(counter_seed,events,counter_observations,combo['finalCombo'])
  # Use the stable per-frame HUD curve for a final, at-most-one-frame timing
  # refinement on supported taps. The hold fitter owns both hold endpoints;
  # moving a hold here would silently change its release time.
  summary['counterTimingRefinement']=align_counter_timing(
   events,
   counter_observations,source_fps,max_shift=1./source_fps,min_improvement=.75)
  # Reconcile after all timing merges so a later guard or spacing pass cannot
  # restore a previous timestamp and leave stale local counts in the package.
  summary['localCounterReconciliation']=reconcile_local_counts(events,allocation)
  if summary['localCounterReconciliation'].get('changed'):
   allocation['localCounterReconciliation']=summary['localCounterReconciliation']

 def refresh_interval_counts():
  count_times=sorted(t for e in events if not e.get('isFake') and not e.get('visualOnly')
                     for t in ([float(e['time'])]+([float(e['time'])+float(e['duration'])]
                         if float(e.get('duration',0))>0 else [])))
  tolerance=0 if allocation['state']=='fitted-video-frame-counter' else .035
  for q in intervals:
   q['estimatedJudgements']=bisect.bisect_right(count_times,float(q['to'])+tolerance+1e-6)-bisect.bisect_right(count_times,float(q['from'])+tolerance+1e-6)
   q['difference']=q['videoJudgements']-q['estimatedJudgements']
  return tolerance

 refresh_interval_counts()
 events.sort(key=lambda e:e['time'])
 events,pruned_noise=prune_visual_noise(events)
 from visual_dedupe import collapse_duplicate_visual_tracks
 summary['duplicateVisualTracks']=collapse_duplicate_visual_tracks(events)
 from hollow_notes_v6 import restore_unrepresented_visual_tracks
 hollow_data=json.loads((root/'hollow-notes-v6.json').read_text())
 summary['hollowVisualRecovery']=restore_unrepresented_visual_tracks(events,hollow_data.get('tracks',[]))
 events.sort(key=lambda e:e['time'])
 (root/'events-v2.json').write_text(json.dumps(events,ensure_ascii=False,indent=2));(root/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2));(root/'dense-counter.json').write_text(json.dumps({'intervals':intervals,'allocation':allocation,'counterTolerance':0 if allocation['state']=='fitted-video-frame-counter' else .035,'extraHitRepairs':len(repairs),'prunedVisualNoise':len(pruned_noise)},ensure_ascii=False,indent=2))
 progress('10/10：导出可游玩的谱面、故事板资源、音频与完整 ZIP')
 duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(video)]))
 source=export_js(events,title,effects=visuals,duration=duration)
 (root/'chart.source.js').write_text(source);(root/'chart.js').write_text(source);package=build_bundle(video,root)
 geometry_review={'state':'skipped-node-unavailable','swaps':[]}
 if shutil.which('node'):
  reviewed=subprocess.run(['node',str(HERE/'candidate_geometry_review.cjs'),str(root)],capture_output=True,text=True)
  if reviewed.returncode:
   raise RuntimeError('rendered duplicate-candidate review failed: '+reviewed.stderr[-1200:])
  result_line=next((line for line in reviewed.stdout.splitlines() if line.startswith('{')),None)
  if result_line:
   geometry_review=json.loads(result_line)
   summary['candidateGeometryReview']=geometry_review
   if geometry_review.get('swaps'):
    events=json.loads((root/'events.json').read_text())
    events.sort(key=lambda e:e['time'])
    (root/'events-v2.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
    source=export_js(events,title,effects=visuals,duration=duration)
    (root/'chart.source.js').write_text(source);(root/'chart.js').write_text(source)
    package=build_bundle(video,root)
 # Geometry review may promote the stronger of two independently tracked
 # capsule passes after the first counter fit. Collapse repeated fast-flow
 # endpoint passes only after that selection, then re-run local allocation so
 # the freed combo slots can be assigned to separately ring-confirmed hits.
 from special_flow_reconcile import collapse_duplicate_hold_endpoint_passes
 summary['specialFlowEndpointReview']=collapse_duplicate_hold_endpoint_passes(events)
 summary['localCounterReconciliationAfterGeometry']=reconcile_local_counts(events,allocation)
 if summary['localCounterReconciliationAfterGeometry'].get('changed'):
  allocation['localCounterReconciliation']=summary['localCounterReconciliationAfterGeometry']
 refresh_interval_counts()
 events.sort(key=lambda e:e['time'])
 (root/'events-v2.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
 (root/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
 (root/'dense-counter.json').write_text(json.dumps({'intervals':intervals,'allocation':allocation,'counterTolerance':0 if allocation['state']=='fitted-video-frame-counter' else .035,'extraHitRepairs':len(repairs),'prunedVisualNoise':len(pruned_noise)},ensure_ascii=False,indent=2))
 source=export_js(events,title,effects=visuals,duration=duration)
 (root/'chart.source.js').write_text(source);(root/'chart.js').write_text(source)
 package=build_bundle(video,root)
 from playability_guard import nudge_early_contacts
 session=gameplay_session(root);playability_changes=[]
 cap=cv2.VideoCapture(str(video));playback_fps=cap.get(cv2.CAP_PROP_FPS) or 30.;cap.release()
 for _ in range(3):
  if not session.get('badNotes') and not session.get('missed') and session.get('combo')==session.get('total'):
   break
  edits=nudge_early_contacts(events,session.get('badNotes',[]),playback_fps)
  if not edits:break
  playability_changes.extend(edits);events.sort(key=lambda e:e['time'])
  (root/'events-v2.json').write_text(json.dumps(events,ensure_ascii=False,indent=2));(root/'events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
  tolerance=refresh_interval_counts()
  (root/'dense-counter.json').write_text(json.dumps({'intervals':intervals,'allocation':allocation,'counterTolerance':tolerance,'extraHitRepairs':len(repairs),'prunedVisualNoise':len(pruned_noise)},ensure_ascii=False,indent=2))
  source=export_js(events,title,effects=visuals,duration=duration)
  (root/'chart.source.js').write_text(source);(root/'chart.js').write_text(source);package=build_bundle(video,root)
  session=gameplay_session(root)
 summary['playabilityGuard']={'state':'passed' if not session.get('badNotes') and not session.get('missed') and session.get('combo')==session.get('total') else 'unresolved','adjustments':playability_changes,'session':session}
 judgements=int(sum(1+int(e['duration']>0) for e in events if not e['isFake']));summary.update(candidates=len(events),judgements=judgements,fakeNotes=int(sum(e['isFake'] for e in events)),extraHitRepairs=len(repairs),prunedVisualNoise=len(pruned_noise),inferredAPNotes=int(sum(e.get('isAlwaysPerfect',False) for e in events if not e['isFake'])),denseCounterIntervals=len(intervals),allocation={k:v for k,v in allocation.items() if k!='cumulative'})
 calibration=json.loads((root/'calibration.json').read_text())
 exported_notes=len(events)-sum(bool(e.get('isFake') and not e.get('path') and e.get('evidence')=='video-counter-and-fresh-source-hit-ring') for e in events)
 report={'nominalPixelsPerSecondAt1280':calibration['nominalPixelsPerSecondAt1280'],'localCounterIntervals':len(intervals),'localCounterUnresolved':sum(abs(q['difference']) for q in intervals),'lineEffects':summary['rails'],'version':7,'automatic':True,'package':package,'source':'input video','visibleCandidates':len(events),'exportedNotes':exported_notes,'videoFinalCombo':combo['finalCombo'],'estimatedRealJudgements':judgements,'inferredFakeNotes':summary['fakeNotes'],'effectOnlyRepairs':len(repairs),'unresolvedCountDifference':combo['finalCombo']-judgements,'reconstruction':summary,'limitations':['遮挡处补检、假音符和 AP 分类仍含推断','粒子由 milplay 的实际击打反馈产生，随机形状不保证逐像素一致','未知的隐藏参数和原始动画组织无法由录像唯一确定']}
 (root/'audit.json').write_text(json.dumps({'videoFinalCombo':combo['finalCombo'],'estimatedJudgements':judgements,'difference':combo['finalCombo']-judgements,'intervals':intervals,'warning':'计数及触摸兼容性不等同于官方原谱逐项一致，遮挡与分类仍含推断。'},ensure_ascii=False,indent=2))
 report['localCounterUnresolved']=sum(abs(q['difference']) for q in intervals)
 (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));progress('完成：V6 谱面、可见判定几何与完整 ZIP 已生成');return report
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',required=True);p.add_argument('--title',default='Video reconstruction');p.add_argument('--resume',action='store_true');a=p.parse_args();print(json.dumps(upgrade(a.video,a.output,a.title,a.resume),ensure_ascii=False),flush=True)
