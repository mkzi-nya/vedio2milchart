import json,sys,subprocess,threading,uuid,shutil
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from urllib.parse import unquote,urlparse
from export_v2 import export_js
from bundle import zip_bytes,minify_source
ROOT=Path(__file__).resolve().parent
JOBS={}
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def reply(self,obj,code=200):
  data=json.dumps(obj,ensure_ascii=False).encode();self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def do_GET(self):
  path=urlparse(self.path).path
  if path.startswith('/api/job/'):
   ident=path.split('/')[-1];state=dict(JOBS.get(ident,{'state':'unknown'}));progress=ROOT/'jobs'/ident/'progress.json'
   if state.get('state')=='running' and progress.exists():
    try:state.update(json.loads(progress.read_text()))
    except (ValueError,OSError):pass
   return self.reply(state)
  if path.endswith('.mp4'):
   file=Path(self.translate_path(path));total=file.stat().st_size if file.is_file() else 0
   if not total:return self.send_error(404)
   start=0;end=total-1;range_header=self.headers.get('Range')
   try:
    if range_header:
     span=range_header.removeprefix('bytes=').split(',')[0];a,b=span.split('-');start=int(a) if a else max(0,total-int(b));end=min(total-1,int(b)) if a and b else total-1
     if not 0<=start<=end<total:return self.send_error(416)
   except ValueError:return self.send_error(416)
   self.send_response(206 if range_header else 200);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
   if range_header:self.send_header('Content-Range',f'bytes {start}-{end}/{total}')
   self.end_headers()
   try:
    with file.open('rb') as f:
     f.seek(start);remaining=end-start+1
     while remaining:
      chunk=f.read(min(1048576,remaining));self.wfile.write(chunk);remaining-=len(chunk)
   except (BrokenPipeError,ConnectionResetError):pass
   return
  return super().do_GET()
 def do_POST(self):
  path=urlparse(self.path).path;size=int(self.headers.get('Content-Length','0'))
  if path=='/api/upload':
   if size<=0 or size>2*1024**3:return self.reply({'error':'视频大小需为 1 字节到 2 GB'},400)
   ident=uuid.uuid4().hex;d=ROOT/'jobs'/ident;d.mkdir(parents=True);video=d/'source.mp4'
   remaining=size
   with video.open('wb') as f:
    while remaining:
     chunk=self.rfile.read(min(1048576,remaining))
     if not chunk:return self.reply({'error':'上传中断'},400)
     f.write(chunk);remaining-=len(chunk)
    try:
     y=float(self.headers.get('X-Judge-Y','.82361'));threshold=int(self.headers.get('X-Threshold','168'))
     if not .2<y<.97 or not 80<=threshold<=245:raise ValueError()
    except ValueError:return self.reply({'error':'校准参数无效'},400)
   title=unquote(self.headers.get('X-Title','Video reconstruction')).strip()[:120] or 'Video reconstruction'
   JOBS[ident]={'state':'running','id':ident,'title':title}
   def run():
    result=subprocess.run([sys.executable,str(ROOT/'automatic.py'),str(video),'--output',str(d),'--title',title],capture_output=True,text=True)
    if result.returncode:JOBS[ident]={'state':'failed','error':result.stderr[-3000:]}
    else:
     JOBS[ident]={'state':'done','base':'/jobs/'+ident,'report':json.loads((d/'report.json').read_text())}
   threading.Thread(target=run,daemon=True).start();return self.reply({'id':ident})
  if path in ('/api/export','/api/export-zip'):
   if size>20*1024**2:return self.reply({'error':'数据过大'},400)
   try:
    data=json.loads(self.rfile.read(size));duration=141.04
    base=(ROOT/str(data.get('base','')).lstrip('/')).resolve()
    if base.is_relative_to(ROOT/'jobs') and (base/'report.json').exists():duration=json.loads((base/'report.json').read_text()).get('package',{}).get('audioDuration',duration)
    src=export_js(data['events'],data.get('title','Video reconstruction'),float(data.get('judgeY',.82361)),data.get('effects',[]),duration)
    src=minify_source(src);payload=src.encode();mime='text/javascript; charset=utf-8'
    if path=='/api/export-zip':
     assetroot=(ROOT/str(data.get('base','')).lstrip('/')).resolve()
     if not assetroot.is_relative_to(ROOT/'jobs'):raise ValueError('需要有效的自动结果资源目录')
     payload=zip_bytes(assetroot,src);mime='application/zip'
    self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
   except (ValueError,KeyError,TypeError,OSError,subprocess.CalledProcessError) as e:self.reply({'error':str(e)},400)
   return
  self.reply({'error':'unknown endpoint'},404)
if __name__=='__main__':
 port=int(sys.argv[1]) if len(sys.argv)>1 else 8765
 print(f'Open http://127.0.0.1:{port}',flush=True);ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
