"""Package only video-derived media and the reconstructed chart for milplay."""
import io,json,os,shutil,subprocess,zipfile
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent

def terser_cli():
    """Find Terser from an override, PATH, or this project's npm install."""
    candidates=[]
    configured=os.environ.get('TERSER_BIN')
    if configured:
        candidates.append(Path(shutil.which(configured) or configured).expanduser())
    on_path=shutil.which('terser')
    if on_path:candidates.append(Path(on_path))
    candidates.extend((ROOT/'node_modules/terser/bin/terser',
                       ROOT.parent/'node_modules/terser/bin/terser'))
    for candidate in candidates:
        if candidate.is_file():return candidate.resolve()
    raise FileNotFoundError('Terser was not found. Run npm install in the project or set TERSER_BIN.')

def minify_source(source):
    result=subprocess.run(['node',str(terser_cli()),'-c','-m','--toplevel'],input=source,text=True,capture_output=True,check=True)
    if len(result.stdout)<100:
        raise ValueError('terser produced an empty chart')
    return result.stdout

def minify_chart(root):
    """Run the same terser pass used for the delivered chart and fail loudly if unavailable."""
    root=Path(root);src=root/'chart.js';tmp=root/'chart.js.terser.tmp'
    terser_cli()
    # Store declarations as numeric command tracks. Decode only at import;
    # the player receives the same native events and no per-frame callbacks.
    from compact_data_v7 import pack
    declarative=root/'chart.declarative.js'
    subprocess.run(['node',str(Path(__file__).with_name('pack_chart_v7.cjs')),str(src),str(declarative)],check=True,capture_output=True)
    packed,stats=pack(declarative.read_text())
    tmp.write_text(minify_source(packed))
    if not tmp.exists() or tmp.stat().st_size<100:
        raise ValueError('terser produced an empty chart')
    tmp.replace(src)
    if src.stat().st_size>1500000:raise ValueError('谱面 JS 超过 1,500,000 字节限制')
    (root/'compression-v7.json').write_text(json.dumps({**stats,'chartBytes':src.stat().st_size,'limitBytes':1500000},indent=2))
    return src.stat().st_size

def zip_bytes(root,chart=None):
    root=Path(root);out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        names=['chart.js','audio.m4a','cover.jpg']
        names.extend(str(p.relative_to(root)) for p in sorted((root/'storyboard').rglob('*')) if p.is_file() and p.name!='rail.png' and not (p.name=='background.jpg' and (root/'storyboard/background.png').exists()))
        for name in names:
            z.writestr(name,chart.encode('utf-8')) if name=='chart.js' and chart is not None else z.write(root/name,name)
    return out.getvalue()

def build_bundle(video,root):
    root=Path(root)
    # The exported chart always references the portable PNG storyboard asset.
    # Older background estimation passes may leave only a JPEG on disk, so
    # normalize that artifact before creating the import package.
    storyboard=root/'storyboard'; png=storyboard/'background.png'; jpg=storyboard/'background.jpg'
    if not png.exists() and jpg.exists():
        with Image.open(jpg) as im:
            im.convert('RGB').save(png,format='PNG',optimize=True)
    minified_size=minify_chart(root)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(video)]))
    stream=next(s for s in info['streams'] if s['codec_type']=='video')
    w,h=int(stream['width']),int(stream['height']);cw=int(min(w,h*.84*16/9))//2*2;ch=int(cw*9/16)//2*2
    # Remove the top HUD from a clean opening frame; retain the video illustration.
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-map','0:a:0','-vn','-c:a','aac','-profile:a','aac_low','-b:a','256k','-movflags','+faststart',str(root/'audio.m4a')],check=True)
    if not ((root/'storyboard/background.png').exists() or (root/'storyboard/background.jpg').exists()):subprocess.run(['ffmpeg','-v','error','-y','-i',str(video),'-frames:v','1','-vf',f'crop={cw}:{ch}:{(w-cw)//2}:{h-ch},scale=1280:720','-q:v','2',str(root/'cover.jpg')],check=True)
    data=zip_bytes(root);temp=root/'milplay.zip.tmp';temp.write_bytes(data);temp.replace(root/'milplay.zip')
    audio=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(root/'audio.m4a')]))
    duration=float(audio['format']['duration'])
    with zipfile.ZipFile(root/'milplay.zip') as z:
        if z.testzip():raise ValueError('ZIP 校验失败')
    return {'file':'milplay.zip','files':zipfile.ZipFile(root/'milplay.zip').namelist(),'audioDuration':duration,'audioCodec':audio['streams'][0]['codec_name'],'audioContainer':'M4A','audioBitrate':256000,'audioOrigin':'视频原音轨编码为 AAC-LC / M4A，保留录制中的击打声','illustrationOrigin':'视频多帧背景估计并去除 HUD；非原始高清曲绘资源' if ((root/'storyboard/background.png').exists() or (root/'storyboard/background.jpg').exists()) else '视频首帧裁切','coverSize':[1280,720],'chartBytes':minified_size,'chartMinifier':'terser','zipCompression':'deflate-9','bytes':len(data)}
