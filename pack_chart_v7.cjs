// Record generated native API commands, then replay declarative data at import.
const fs=require('node:fs'),vm=require('node:vm');
const input=process.argv[2],output=process.argv[3];
const meta={},bpms=[],lines=[],notes=[],storyboards=[],tracks=new Map();
const m={withProperty(k,v){meta[k]=v;return m},env(){return 1.66},timing(...args){bpms.push(args);return bpms.length-1},line(){lines.push({});return lines.length-1},note(...args){notes.push(args);return notes.length-1},storyboardObject(...args){storyboards.push(args);return storyboards.length-1},animation(b,from,to,k,fv,tv,d,o,press,ease,expr,custom){if(b!==0||press||ease||expr||custom)throw Error('Unsupported packed animation');const key=[d,o,k].join(',');if(!tracks.has(key))tracks.set(key,[]);tracks.get(key).push([from,to,fv,tv]);}};
// Dense video reconstructions contain many short native animation tracks.
// Give the import-time capture enough headroom on slower hosts instead of
// failing midway through an otherwise valid large chart.
vm.runInNewContext(fs.readFileSync(input,'utf8'),{MilizeBeatmap:m},{timeout:60000});
let source=`const m=MilizeBeatmap;const fc=1.66/(Number(m.env("user.flow_speed"))||1.66);`;
for(const [k,v] of Object.entries(meta))source+=`m.withProperty(${JSON.stringify(k)},${JSON.stringify(v)});`;
for(const b of bpms)source+=`m.timing(${b.join(',')});`;
source+=`for(let i=0;i<${lines.length};i++)m.line();`;
for(const s of storyboards)source+=`m.storyboardObject(${s.map(JSON.stringify).join(',')});`;
source+=`for(const q of ${JSON.stringify(notes.map(q=>q.map(v=>typeof v==='boolean'?+v:v)))})m.note(q[0],q[1],q[2],q[3],q[4],!!q[5],!!q[6]);`;
// Explicit absolute endpoints preserve discontinuities, gaps and event order.
source+='function __track(d,o,k,rows){for(const r of rows){let f=k===5&&d===1?fc:1;m.animation(0,r[0],r[1],k,r[2]*f,r[3]*f,d,o,0,0,false,"");}}';
for(const [key,rows] of tracks)source+=`__track(${key},${JSON.stringify(rows)});`;
fs.writeFileSync(output,source);
console.log(JSON.stringify({lines:lines.length,notes:notes.length,tracks:tracks.size,animations:[...tracks.values()].reduce((a,b)=>a+b.length,0),declarativeBytes:Buffer.byteLength(source)}));
