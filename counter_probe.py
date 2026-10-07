"""Read the original counter at dense fixed sample times, with independent local OCR."""
import cv2,numpy as np,json,subprocess,concurrent.futures,hashlib
from pathlib import Path

def scan(video,root):
 root=Path(root);cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS) or 30;f=0;jobs=[];cache={}
 while True:
  ok,raw=cap.read()
  if not ok:break
  if f%3==0:
   if raw.shape[:2]!=(720,1280):raw=cv2.resize(raw,(1280,720))
   im=raw[62:126,510:770].min(2);railmask=np.zeros_like(im);railmask[:,(im>150).sum(0)>48]=255
   im=cv2.inpaint(im,railmask,3,cv2.INPAINT_TELEA);mask=im
   # OCR accepts an encoded source crop; nothing is uploaded.
   key=hashlib.sha256(mask.tobytes()).hexdigest();ok,png=cv2.imencode('.png',cv2.resize(mask,(1040,256)));jobs.append((f/fps,key,png.tobytes()))
  f+=1
 cap.release()
 def read(job):
  t,key,png=job
  if key in cache:return t,*cache[key]
  r=subprocess.run(['tesseract','stdin','stdout','--psm','7','-c','tessedit_char_whitelist=0123456789','tsv'],input=png,capture_output=True)
  words=[];confs=[]
  for line in r.stdout.decode(errors='ignore').splitlines()[1:]:
   fields=line.split('\t')
   if len(fields)>=12 and fields[11].isdigit():words.append(fields[11]);confs.append(float(fields[10]))
  label=int(''.join(words)) if words else None;conf=min(confs) if confs else 0;cache[key]=(label,conf);return t,label,conf
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for i,q in enumerate(pool.map(read,jobs)):
   results.append({'time':q[0],'combo':q[1],'confidence':q[2]})
   if i%300==299:print('Dense OCR:',i+1,'/',len(jobs),flush=True)
 (root/'counter-probes.json').write_text(json.dumps(results,indent=2));print('Counter probes:',len(results),'unique',len(cache),flush=True);return results
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--output',required=True);a=p.parse_args();scan(a.video,a.output)
