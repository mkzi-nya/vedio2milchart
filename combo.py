"""Read combo counter solely from source video using optional local Tesseract."""
import subprocess, shutil, json, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
P=argparse.ArgumentParser();P.add_argument('video');P.add_argument('--output',required=True);args=P.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
ocr=shutil.which('tesseract')
if not ocr:raise SystemExit('未找到 tesseract，可跳过连击校验')
# Learn digit shapes from labelled, video-derived examples, then classify every frame.
p=subprocess.Popen(['ffmpeg','-v','error','-i',args.video,'-vf','fps=30:start_time=0:round=up,scale=640:360,crop=130:32:255:31','-f','rawvideo','-pix_fmt','gray','-'],stdout=subprocess.PIPE)
frames=[]
while True:
 raw=p.stdout.read(130*32)
 if len(raw)!=130*32:break
 im=np.frombuffer(raw,np.uint8).reshape(32,130).copy();im[:,(im>175).sum(axis=0)>25]=0;frames.append(im)
p.wait()
def glyphs(im):
 mask=im>175
 # No ALL PERFECT text in this crop. Separate each digit by blank columns.
 cols=np.flatnonzero(mask.sum(axis=0)>0)
 if not len(cols):return []
 groups=np.split(cols,np.where(np.diff(cols)>1)[0]+1);res=[]
 for g in groups:
  if len(g)<2:continue
  crop=mask[:,g[0]:g[-1]+1];ys=np.flatnonzero(crop.sum(axis=1))
  if not len(ys):continue
  crop=crop[ys[0]:ys[-1]+1]
  if crop.shape[0]<6:continue
  norm=np.asarray(Image.fromarray(crop.astype('uint8')*255).resize((16,24)))/255
  res.append(norm)
 return res
bank={str(i):[] for i in range(10)};labels=[]
for i in range(0,len(frames),120):
 f=out/'combo-tmp.png';Image.fromarray(frames[i]).resize((520,128)).save(f)
 r=subprocess.run([ocr,str(f),'stdout','--psm','7','-c','tessedit_char_whitelist=0123456789'],capture_output=True,text=True)
 label=r.stdout.strip();gs=glyphs(frames[i])
 if label.isdigit() and len(label)==len(gs):
  labels.append([i/30,int(label)])
  for d,g in zip(label,gs):bank[d].append(g)
f.unlink(missing_ok=True)
for d in bank:
 if not bank[d]:print('missing digit',d)
raw=[]
for i,im in enumerate(frames):
 gs=glyphs(im)
 if not gs:continue
 ds=[];err=0
 for g in gs:
  scores={d:min((np.mean((g-q)**2) for q in qs),default=1) for d,qs in bank.items()}
  digit=min(scores,key=scores.get);ds.append(digit);err=max(err,scores[digit])
 value=int(''.join(ds))
 if err<.16:raw.append([i,value,round(float(err),4)])
# Counter should increase monotonically in an AP playthrough. Require consecutive frames.
confirmed=[];last=-1;lastframe=0
for j,(frame,value,err) in enumerate(raw):
 if j+1>=len(raw):break
 nxt=raw[j+1]
 if nxt[0]!=frame+1 or nxt[1]!=value:continue
 if value>=last and value-last<=max(30,(frame-lastframe)/30*35):
  if value!=last:confirmed.append({'time':frame/30,'combo':value,'error':err})
  last=value;lastframe=frame
(out/'combo.json').write_text(json.dumps({'samples':confirmed,'ocrLabels':labels,'finalCombo':last},ensure_ascii=False,indent=2))
print(json.dumps({'comboChanges':len(confirmed),'finalCombo':last,'labelCount':len(labels),'digits':{d:len(v) for d,v in bank.items()}}))
