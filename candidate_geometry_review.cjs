// Resolve alternate detections of one measured track by checking their actual
// Milplay-rendered head and tail against the source-video trajectory.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const canvas = require('@napi-rs/canvas');

const outputDir = path.resolve(process.argv[2] || '');
if (!process.argv[2]) throw new Error('Usage: node candidate_geometry_review.cjs <output-directory>');
const eventsPath = path.join(outputDir, 'events.json');
const chartPath = path.join(outputDir, 'chart.js');
const events = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
const milplayRoot = path.resolve(__dirname, '../milplay');

let harnessSource = fs.readFileSync(path.join(milplayRoot, 'tests/harness.js'), 'utf8');
harnessSource = harnessSource.replace(/    getContext\(\)\{[^\n]+\}/,
  `    getContext(){if(!this.__nativeCanvas)this.__nativeCanvas=opts.canvas.createCanvas(this.width,this.height);const g=this.__nativeCanvas.getContext('2d');return new Proxy(g,{get:(o,k)=>k==='drawImage'?((im,...a)=>o.drawImage(im.__nativeCanvas||im,...a)):typeof o[k]==='function'?o[k].bind(o):o[k],set:(o,k,v)=>(o[k]=v,true)})}`);
harnessSource = harnessSource.replace(/  class ImageFake[^\n]+/,
  `  class ImageFake extends opts.canvas.Image {constructor(){super();images.push(this)}get naturalWidth(){return this.width}get naturalHeight(){return this.height}get complete(){return this.width>0}ready(){}set src(s){this._source=s;if(s.startsWith('assets/')&&fs.existsSync(path.join(root,s)))super.src=fs.readFileSync(path.join(root,s));else if(s.startsWith('data:'))super.src=s;}get src(){return this._source}}`);
const harnessModule = { exports: {} };
vm.runInThisContext('(function(require,module,exports){' + harnessSource + '\n})')(
  require, harnessModule, harnessModule.exports);

function quantile(values, p) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))] : null;
}

function measure(event, note, runtime, lineTransform, noteFrame) {
  const compare = (samples, useTail) => {
    const errors = [];
    for (const point of samples || []) {
      const time = Number(point[0]);
      const line = lineTransform(runtime, note.lineIdx, time, 1280, 720);
      const frame = noteFrame(runtime, note, time, line, 1280, 720);
      if (!frame || !(frame.alpha > .001)) continue;
      const rendered = useTail ? frame.tail : frame.center;
      errors.push(Math.hypot(rendered.x - Number(point[1]), rendered.y - Number(point[2])));
    }
    return { count: errors.length, median: quantile(errors, .5), p90: quantile(errors, .9) };
  };
  const head = compare(event.path, false);
  const tail = Number(event.duration || 0) > 0 ? compare(event.tailPath, true) : { count: 0, median: null, p90: 0 };
  if (head.count < 8 || (Number(event.duration || 0) > 0 && tail.count < 8)) return null;
  return {
    head, tail,
    score: head.p90 + .8 * (tail.p90 || 0),
  };
}

(async () => {
  const harness = harnessModule.exports.createHarness(milplayRoot, { canvas });
  harness.context.src = fs.readFileSync(chartPath, 'utf8');
  const parsed = await harness.run('parseText(src,"chart.js")');
  harness.context.chart = parsed.chart;
  harness.run('innerWidth=1280;innerHeight=720;devicePixelRatio=1;resizeCanvas=()=>{};state.viewScale=1;state.panX=state.panY=0;state.appMode="play";state.flowSpeed=1.66;state.runtime=makeRuntime(chart);state.chart=chart;state.duration=state.runtime.duration;');
  const runtime = harness.run('state.runtime');
  const lineTransform = harness.run('transformLine');
  const noteFrame = harness.run('__pluNoteFrame');

  const evaluated = events.map((event, index) => {
    if (event.track == null || !Array.isArray(event.path) || event.path.length < 8) return { index, event, fit: null };
    const hold = Number(event.duration || 0) > 0;
    const notes = runtime.notes.filter(note =>
      Math.abs(note.startSec - Number(event.time)) < .04 &&
      Number(note.type) === Number(event.type) &&
      Boolean(note.isFake) === Boolean(event.isFake) &&
      Boolean(note.isHold) === hold);
    let best = null;
    for (const note of notes) {
      const fit = measure(event, note, runtime, lineTransform, noteFrame);
      if (!fit) continue;
      const score = fit.score + Math.abs(note.startSec - Number(event.time)) * .25;
      if (!best || score < best.score) best = { note, ...fit, score };
    }
    return { index, event, fit: best };
  });

  const groups = new Map();
  for (const row of evaluated) {
    if (!row.fit) continue;
    const event = row.event;
    const key = `${event.track}|${event.targetTrack ?? '-'}|${event.type}|${Number(event.duration || 0) > 0}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }

  const swaps = [], comparisons = [];
  for (const [key, rows] of groups) {
    rows.sort((a, b) => Number(a.event.time) - Number(b.event.time));
    for (let i = 0; i < rows.length; i++) {
      const cluster = rows.filter(row =>
        Math.abs(Number(row.event.time) - Number(rows[i].event.time)) <= .01 &&
        Math.abs(Number(row.event.duration || 0) - Number(rows[i].event.duration || 0)) <= .02);
      if (cluster.length < 2) continue;
      if (rows[i].index !== Math.min(...cluster.map(row => row.index))) continue;
      const real = cluster.filter(row => !row.event.isFake).sort((a, b) => a.fit.score - b.fit.score);
      const fake = cluster.filter(row => row.event.isFake).sort((a, b) => a.fit.score - b.fit.score);
      if (!real.length || !fake.length) continue;
      const current = real[0], alternate = fake[0];
      const improvement = current.fit.score - alternate.fit.score;
      comparisons.push({
        group: key,
        time: Number(alternate.event.time),
        realIndex: current.index,
        fakeIndex: alternate.index,
        realScore: current.fit.score,
        fakeScore: alternate.fit.score,
        improvement,
      });
      if (alternate.fit.score > 15 || improvement < 8 || alternate.fit.score > current.fit.score * .5) continue;
      current.event.isFake = true;
      alternate.event.isFake = false;
      current.event.geometryReviewRole = 'duplicate-track-alternate';
      alternate.event.geometryReviewRole = 'trajectory-fit-winner';
      swaps.push({
        group: key,
        time: Number(alternate.event.time),
        track: alternate.event.track,
        previousRealIndex: current.index,
        selectedIndex: alternate.index,
        previousRealFit: { headP90: current.fit.head.p90, tailP90: current.fit.tail.p90, score: current.fit.score },
        selectedFit: { headP90: alternate.fit.head.p90, tailP90: alternate.fit.tail.p90, score: alternate.fit.score },
      });
    }
  }

  if (swaps.length) fs.writeFileSync(eventsPath, JSON.stringify(events, null, 2));
  const report = { state: 'reviewed-rendered-duplicate-trajectories', evaluatedTrackCandidates: evaluated.filter(row => row.fit).length, comparisons, swaps };
  fs.writeFileSync(path.join(outputDir, 'geometry-candidate-review.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
})().catch(error => { console.error(error); process.exitCode = 1; });
