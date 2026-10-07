"""Pack numeric animation matrices once at import; playback stays native.

Decimal integers are lossless for authored samples. Unrounded floats use IEEE
754, so user-flow scaling and millisecond visibility boundaries are preserved.
The small LZ77 decoder is self contained: no eval, network or runtime plugin.
"""
import re,json,struct,base64,math
from collections import defaultdict,deque

MATRIX=re.compile(r'\[\[(?:[-+0-9.eE,\[\]])+\]\]')

def uint(buf,v):
 while v>=128:buf.append((v&127)|128);v>>=7
 buf.append(v)

def lz77(data):
 out=bytearray();index=defaultdict(lambda:deque(maxlen=12));i=0
 while i<len(data):
  flag_at=len(out);out.append(0);flag=0
  for bit in range(8):
   if i>=len(data):break
   length=0;offset=0;key=data[i:i+3]
   if len(key)==3:
    for old in reversed(index.get(key,())):
     if i-old>65535:break
     n=3;limit=min(258,len(data)-i)
     while n<limit and data[old+n]==data[i+n]:n+=1
     if n>length:length=n;offset=i-old
     if n==limit:break
   if length>=4:
    flag|=1<<bit;out.extend((length-3,offset&255,offset>>8));stop=i+length
   else:out.append(data[i]);stop=i+1
   while i<stop:
    if i+3<=len(data):index[data[i:i+3]].append(i)
    i+=1
  out[flag_at]=flag
 return bytes(out)

DECODER=r'''const __packed=(()=>{const s=PAYLOAD,chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",lookup=new Int16Array(128);for(let i=0;i<64;i++)lookup[chars.charCodeAt(i)]=i;let v=0,bits=0,p=0;const z=new Uint8Array(Math.floor(s.length*3/4));for(let i=0;i<s.length;i++){if(s[i]==='=')break;v=(v<<6)|lookup[s.charCodeAt(i)];bits+=6;if(bits>=8){bits-=8;z[p++]=(v>>bits)&255;}}const out=new Uint8Array(LENGTH);let a=0,o=0;while(a<p){let flags=z[a++];for(let bit=0;bit<8&&a<p;bit++){if(flags&(1<<bit)){let n=z[a++]+3,d=z[a++]|(z[a++]<<8);for(let j=0;j<n;j++){out[o]=out[o-d];o++;}}else out[o++]=z[a++];}}if(o!==out.length)throw Error('Chart data length mismatch');return out;})();let __cursor=0;const __view=new DataView(__packed.buffer);function __u(){let n=0,f=1,c;do{c=__packed[__cursor++];n+=(c&127)*f;f*=128;}while(c&128);return n;}function __matrix(){const n=__u(),d=__packed[__cursor++],scales=[],last=Array(d).fill(0),rows=[];for(let k=0;k<d;k++)scales.push(__packed[__cursor++]);for(let i=0;i<n;i++){let q=[];for(let k=0;k<d;k++){if(scales[k]===255){q.push(__view.getFloat64(__cursor,true));__cursor+=8;}else{let z=__u();let predict=last[k];if(d===4){let other=k<2?(k===0?1:0):(k===2?3:2);if(scales[other]!==255)predict=Math.round(last[other]*10**(scales[k]-scales[other]));}last[k]=predict+(z%2?-(z+1)/2:z/2);q.push(last[k]/10**scales[k]);}}rows.push(q);}return rows;}'''

def pack(source):
 tape=bytearray();count=0
 def replace(match):
  nonlocal count
  try:rows=json.loads(match[0])
  except ValueError:return match[0]
  if not rows or not all(isinstance(q,list) and len(q)==len(rows[0]) and all(isinstance(v,(int,float)) for v in q) for q in rows):return match[0]
  d=len(rows[0]);scales=[]
  for k in range(d):
   scale=next((e for e in range(6) if all(round(q[k]*10**e)/10**e==q[k] for q in rows)),255);scales.append(scale)
  uint(tape,len(rows));tape.append(d);tape.extend(scales);last=[0]*d
  for q in rows:
   for k,v in enumerate(q):
    if scales[k]==255:tape.extend(struct.pack('<d',v));continue
    iv=round(v*10**scales[k]);predict=last[k]
    if d==4:
     other=(1 if k==0 else 0) if k<2 else (3 if k==2 else 2)
     if scales[other]!=255:predict=math.floor(last[other]*10**(scales[k]-scales[other])+.5)
    delta=iv-predict;last[k]=iv;uint(tape,2*delta if delta>=0 else -2*delta-1)
  count+=1;return '__matrix()'
 body=MATRIX.sub(replace,source)
 if not count:return source,{'matrices':0}
 compressed=lz77(bytes(tape));encoded=base64.b64encode(compressed).decode()
 header=DECODER.replace('PAYLOAD',json.dumps(encoded)).replace('LENGTH',str(len(tape)))
 return header+'\n'+body,{'matrices':count,'numericBytes':len(tape),'compressedNumericBytes':len(compressed),'codec':'decimal-delta/IEEE754 + LZ77 + base64; import-only'}
