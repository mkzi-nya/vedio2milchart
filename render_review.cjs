// Render through milplay's real scripts with a native Canvas2D backend, no browser automation.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const canvas=require('@napi-rs/canvas');
const {execFileSync}=require('node:child_process');
const sourceVideo=process.argv[6]||process.env.SOURCE_VIDEO;
if(!sourceVideo)throw new Error('Usage: node render_review.cjs <converted-output-directory> [times] [render-directory] [WIDTHxHEIGHT] <source-video>, or set SOURCE_VIDEO.');
const videoInfo=JSON.parse(execFileSync('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',sourceVideo],{encoding:'utf8'})).streams[0];
const dimensions=(process.argv[5]||`${videoInfo.width}x${videoInfo.height}`).split('x').map(Number);
const renderWidth=dimensions[0],renderHeight=dimensions[1];
if(!renderWidth||!renderHeight)throw Error('Invalid render dimensions');
let source=fs.readFileSync(path.join(__dirname,'../milplay/tests/harness.js'),'utf8');
source=source.replace(/    getContext\(\)\{[^\n]+\}/,`    getContext(){if(!this.__nativeCanvas)this.__nativeCanvas=opts.canvas.createCanvas(this.width,this.height);const g=this.__nativeCanvas.getContext('2d');return new Proxy(g,{get:(o,k)=>k==='drawImage'?((im,...a)=>o.drawImage(im.__nativeCanvas||im,...a)):typeof o[k]==='function'?o[k].bind(o):o[k],set:(o,k,v)=>(o[k]=v,true)})}`);
source=source.replace(/  class ImageFake[^\n]+/,`  class ImageFake extends opts.canvas.Image {constructor(){super();images.push(this)}get naturalWidth(){return this.width}get naturalHeight(){return this.height}get complete(){return this.width>0}ready(){}set src(s){this._source=s;if(s.startsWith('assets/')&&fs.existsSync(path.join(root,s)))super.src=fs.readFileSync(path.join(root,s));else if(s.startsWith('data:'))super.src=s;}get src(){return this._source}}`);
source=source.replace('  for(const m of html.matchAll(/<script src="([^"]+)"/g)){','  vm.runInContext("(()=>{let s=0x51f15eed;Math.random=()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296}})()",context);\n  for(const m of html.matchAll(/<script src="([^"]+)"/g)){');
let moduleObject={exports:{}};vm.runInThisContext('(function(require,module,exports){'+source+'\n})')(require,moduleObject,moduleObject.exports);
(async()=>{let root=path.resolve(__dirname,'../milplay'),folder=process.argv[2];if(!folder)throw Error('Usage: node render_review.cjs <converted-output-directory> [times] [render-directory] [WIDTHxHEIGHT] <source-video>');let h=moduleObject.exports.createHarness(root,{canvas});h.context.src=fs.readFileSync(path.resolve(__dirname,folder,'chart.js'),'utf8');let parsed=await h.run('parseText(src,"chart.js")');h.context.chart=parsed.chart;h.context.bg=await canvas.loadImage(path.resolve(__dirname,folder,'cover.jpg'));
for(let name of fs.readdirSync(path.join(root,'assets')).filter(x=>x.endsWith('.png'))){h.context.img=await canvas.loadImage(path.join(root,'assets',name));h.context.key=name.replace('.png','');h.run('state.images[key]=img;if(key.startsWith("rain_"))__rainHoldImgs.set(key.slice(5),img)')}
// Uploaded storyboard artwork is loaded using exactly its referenced resource names.
if(fs.existsSync(path.resolve(__dirname,folder,'storyboard'))){for(let name of fs.readdirSync(path.resolve(__dirname,folder,'storyboard'))){h.context.im=await canvas.loadImage(path.resolve(__dirname,folder,'storyboard',name));h.context.name='storyboard/'+name;h.run('storyCache.set(name,{drawable:im,sourceWidth:im.width,sourceHeight:im.height})')}}
h.context.reviewWidth=renderWidth;h.context.reviewHeight=renderHeight;
h.get('stage').width=renderWidth;h.get('stage').height=renderHeight;
if(h.get('stage').__nativeCanvas){h.get('stage').__nativeCanvas.width=renderWidth;h.get('stage').__nativeCanvas.height=renderHeight;}
h.run('innerWidth=reviewWidth;innerHeight=reviewHeight;devicePixelRatio=1;resizeCanvas=()=>{};state.backgroundImage=bg;state.bgBrightness=1;state.appMode="play";state.viewScale=1;state.panX=state.panY=0;state.runtime=makeRuntime(chart);state.chart=chart;state.duration=state.runtime.duration;__gpTest.fresh(state.runtime);__gpTest.gp.autoplay=true;state.autoplay=true;state.hudVisible=false;drawCombo=()=>{};drawOverlay=()=>{};');
if(process.env.REVIEW_TRACE_NOTES)h.run('window.__reviewDrawNotes=[];window.__reviewDrawNoteBase=drawNote;drawNote=function(rt,n,sec,st,w,h){const f=__pluNoteFrame(rt,n,sec,st,w,h);if(f&&f.center.x>=0&&f.center.x<=reviewWidth&&f.center.y>=-80&&f.center.y<=reviewHeight+80)window.__reviewDrawNotes.push({key:n.key,globalIdx:n.globalIdx,lineIdx:n.lineIdx,startSec:n.startSec,endSec:n.endSec,type:n.type,isHold:n.isHold,isFake:n.isFake,center:f.center,tail:f.tail,visualW:f.visualW,floorHead:f.floorHead,floorTail:f.floorTail,alpha:f.alpha,baseX:f.baseX,baseY:f.baseY,noteRotation:f.noteRot,lineCenter:st.center,lineRotation:st.rotation,lineScale:st.scale,lineFlow:st.flow,raw:n.note});return window.__reviewDrawNoteBase(rt,n,sec,st,w,h)}');
let times=(process.argv[3]||'2.6,5,20,40,60,80,90,100,120,136').split(',').map(Number);let out=path.resolve(__dirname,process.argv[4]||'review');fs.mkdirSync(out,{recursive:true});let timings=[];
let dense=process.env.REVIEW_DENSE,sourceRGB,sourceHudPacked,sourceSettings,small,smallCtx,metrics=[],denseFrames=[];
if(dense){sourceSettings=JSON.parse(fs.readFileSync(path.join(out,'source-settings.json')));sourceRGB=fs.readFileSync(path.join(out,'source-rgb-320.bin'));sourceHudPacked=fs.readFileSync(path.join(out,'source-hud-mask-320.bin'));denseFrames=Array.from({length:sourceSettings.frames},(_,i)=>i);if(process.env.REVIEW_INTERVALS){const ranges=process.env.REVIEW_INTERVALS.split(',').map(s=>s.split('-').map(Number));denseFrames=denseFrames.filter(i=>ranges.some(([a,b])=>i/sourceSettings.fps>=a&&i/sourceSettings.fps<=b));}times=denseFrames.map(i=>i/sourceSettings.fps);small=canvas.createCanvas(320,180);smallCtx=small.getContext('2d');}
const compareFrames=new Set((process.env.REVIEW_DIFF_FRAMES||'').split(',').filter(Boolean).map(Number));
const compareOut=dense&&process.env.REVIEW_DIFF_DIR?path.resolve(__dirname,process.env.REVIEW_DIFF_DIR):null;
const renderTimeOffset=Number(process.env.REVIEW_TIME_OFFSET_SEC)||0;
if(compareOut)fs.mkdirSync(compareOut,{recursive:true});
// The native-resolution UI mask is prepared once from each source frame and
// packed to disk. Use the same mask for metrics, line geometry and previews.
const hudMaskFrameBytes=Math.ceil(320*180/8);
function sourceHudMaskForFrame(frameIndex){const start=frameIndex*hudMaskFrameBytes,packed=sourceHudPacked.subarray(start,start+hudMaskFrameBytes);if(packed.length!==hudMaskFrameBytes)throw Error('Missing source UI mask frame '+frameIndex);const ignored=new Uint8Array(320*180);for(let i=0;i<ignored.length;i++)ignored[i]=(packed[i>>3]>>(i&7))&1;return ignored;}
function mask(rgb,stride,ignored){const m=new Uint8Array(320*180);for(let y=0;y<180;y++)for(let x=0;x<320;x++){let i=y*320+x;if(ignored[i])continue;let k=i*stride,r=rgb[k],g=rgb[k+1],b=rgb[k+2],lo=Math.min(r,g,b),hi=Math.max(r,g,b);if(lo>58&&hi-lo<26)m[i]=1;}return m;}
function lineDifference(a,b,ignored){let missing=0,extra=0,na=0,nb=0;for(let y=0;y<180;y++)for(let x=0;x<320;x++){let i=y*320+x;if(ignored[i]||(!a[i]&&!b[i]))continue;let an=0,bn=0;for(let yy=-1;yy<=1;yy++)for(let xx=-1;xx<=1;xx++){const xx2=x+xx,yy2=y+yy;if(xx2<0||xx2>=320||yy2<0||yy2>=180||ignored[yy2*320+xx2])continue;an+=a[yy2*320+xx2];bn+=b[yy2*320+xx2];}if(a[i]){na++;if(!bn)missing++;}if(b[i]){nb++;if(!an)extra++;}}return{sourceBrightNeutralPixels:na,renderBrightNeutralPixels:nb,missingSourcePixels:missing,extraRenderPixels:extra,neutralMismatch:missing+extra};}
function saveFilteredCompare(original,rendered,ignored,frameIndex,time){
 if(!compareOut||!compareFrames.has(frameIndex))return;
 const headerHeight=24,triptych=canvas.createCanvas(960,180+headerHeight),tc=triptych.getContext('2d');
 tc.fillStyle='#141414';tc.fillRect(0,0,960,headerHeight);tc.fillStyle='#f0f0f0';tc.font='10px sans-serif';
 tc.fillText(`${time.toFixed(3)}s · SOURCE · UI PIXELS REPLACED`,6,16);
 tc.fillText('MILPLAY · HUD OFF',326,16);
 tc.fillText('DIFFERENCE ×3 · HATCH = IGNORED UI',646,16);
 for(let panel=0;panel<3;panel++){
  const c=canvas.createCanvas(320,180),cx=c.getContext('2d'),im=cx.createImageData(320,180);
 for(let i=0;i<320*180;i++){
   const s=i*3,d=i*4,x=i%320,y=(i/320)|0;
   for(let ch=0;ch<3;ch++){
    if(panel===2){
     const ignoredHatch=((x+y)%8<2)||((x-y+800)%8<2);
     im.data[d+ch]=ignored[i]?(ignoredHatch?[68,72,88][ch]:[30,32,42][ch]):Math.min(255,Math.abs(original[s+ch]-rendered[d+ch])*3);
    }else im.data[d+ch]=(panel===0&&!ignored[i])?original[s+ch]:rendered[d+ch];
   }
   im.data[d+3]=255;
  }
  cx.putImageData(im,0,0);tc.drawImage(c,panel*320,headerHeight);
 }
 fs.writeFileSync(path.join(compareOut,'compare-'+String(frameIndex).padStart(5,'0')+'-'+time.toFixed(3)+'.png'),triptych.toBuffer('image/png'));
}
for(let fi=0;fi<times.length;fi++){let t=times[fi]+renderTimeOffset;h.context.t=t;if(process.env.REVIEW_TRACE_NOTES)h.run('window.__reviewDrawNotes.length=0');const began=performance.now();h.run('state.currentTime=t;render()');timings.push({time:t,renderMs:performance.now()-began});if(process.env.REVIEW_TRACE_NOTES)fs.writeFileSync(path.join(out,'drawn-notes-'+t+'.json'),h.run('JSON.stringify(window.__reviewDrawNotes||[])'));if(process.env.REVIEW_TRACE_LINES){const lines=h.run('JSON.stringify(Array.from({length:chart.lines.length},(_,i)=>{const s=transformLine(state.runtime,i,t,reviewWidth,reviewHeight);return {index:i,center:s?.center,rotation:s?.rotation,scale:s?.scale,flow:s?.flow,transparency:s?.transparency,bodyAlpha:s?.bodyAlpha,headAlpha:s?.headAlpha,wholeAlpha:s?.wholeAlpha,visible:s?.visible}}))');fs.writeFileSync(path.join(out,'line-states-'+t+'.json'),lines);}const surface=h.get('stage').__nativeCanvas;if(surface.width!==renderWidth||surface.height!==renderHeight)throw Error('Render resolution mismatch');if(dense){smallCtx.drawImage(surface,0,0,320,180);const rendered=smallCtx.getImageData(0,0,320,180).data,frameIndex=denseFrames[fi];const original=sourceRGB.subarray(frameIndex*320*180*3,(frameIndex+1)*320*180*3);const ignored=sourceHudMaskForFrame(frameIndex);let error=0,pixels=0,topError=0,topPixels=0,bottomError=0,bottomPixels=0,ignoredPixels=0;for(let y=0;y<180;y++)for(let x=0;x<320;x++){let i=y*320+x;if(ignored[i]){ignoredPixels++;continue;}let pxError=Math.abs(original[i*3]-rendered[i*4])+Math.abs(original[i*3+1]-rendered[i*4+1])+Math.abs(original[i*3+2]-rendered[i*4+2]);error+=pxError;pixels++;if(y<90){topError+=pxError;topPixels++;}else{bottomError+=pxError;bottomPixels++;}}metrics.push({frame:frameIndex,time:t,sourceTime:frameIndex/sourceSettings.fps,meanAbsRGBFilteredUI:error/(pixels*3),meanAbsRGBTopPlayfield:topError/(topPixels*3),meanAbsRGBLowerPlayfield:bottomError/(bottomPixels*3),comparedPixels:pixels,filteredHudPixels:ignoredPixels,topPlayfieldPixels:topPixels,lowerPlayfieldPixels:bottomPixels,...lineDifference(mask(original,3,ignored),mask(rendered,4,ignored),ignored)});saveFilteredCompare(original,rendered,ignored,frameIndex,t);if(fi%300===0)console.log('All-frame QA',fi+'/'+times.length);}
else fs.writeFileSync(path.join(out,'milplay-'+t+'.png'),surface.toBuffer('image/png'));if(process.env.REVIEW_TRACE){fs.writeFileSync(path.join(out,'anchors-'+t+'.json'),JSON.stringify(h.run('state.runtime.notes.filter(n=>!n.isFake&&n.startSec<=t&&n.startSec>t-.5).map(n=>({id:n.animIdx,time:n.startSec,anchor:((a)=>({x:a?.x,y:a?.y,alpha:a?.alpha}))(__pluAnchorEffectNote(n,n.startSec,reviewWidth,reviewHeight,state.runtime))}))'),null,2));}}
if(dense)fs.writeFileSync(path.join(out,'all-frame-metrics.json'),JSON.stringify(metrics));fs.writeFileSync(path.join(out,'render-settings.json'),JSON.stringify({sourceWidth:videoInfo.width,sourceHeight:videoInfo.height,canvasWidth:h.get('stage').__nativeCanvas.width,canvasHeight:h.get('stage').__nativeCanvas.height,devicePixelRatio:1,viewScale:1,panX:0,panY:0,mode:'play',autoplay:true,hudRendered:false,hudLayersSuppressed:['combo','canvas overlay'],domUiComposited:false,effectSeed:'xorshift32:0x51f15eed',renderTimeOffset,sourceHudMask:'native-resolution glyph/stroke separation; native UI pixels substituted and excluded; no whole HUD boxes hidden; top playfield retained; ignored pixels hatched in diff',timings},null,2));console.log('Rendered',times.length,'frames at',renderWidth+'x'+renderHeight,'with the real milplay renderer');})().catch(e=>{console.error(e);process.exitCode=1});
