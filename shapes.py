"""Reject thin rail/particle shapes misclassified as filled Hold bodies."""
import json
import numpy as np

def normalize_holds(events,passes):
 tables={str(k):{t['id']:t['samples'] for t in json.loads((passes/str(k)/'tracks.json').read_text())} for k in (168,150,188)}
 rejected=0
 for e in events:
  if e['duration']<=0 or str(e.get('pass')) not in tables:continue
  s=np.array(tables[str(e['pass'])][e['track']]);body=s[s[:,6]>0]
  if len(body)>=3 and float(np.median(body[:,7]))<.45:
   e['duration']=0;normal=s[s[:,6]==0];e['type']=int(np.median(normal[:,5])>=.5) if len(normal) else 0;e['holdEvidence']='thin-rail-or-effect-body';rejected+=1
 return rejected
