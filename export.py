import json, math

def export_js(events,title='Video reconstruction',judge_y=.82361,effects=None):
    out=['// Video-only estimated reconstruction. Time origin = source video 0.','const m = MilizeBeatmap;',f'm.withProperty("Title",{json.dumps(title,ensure_ascii=False)}).withProperty("Difficulty","Cloudburst").withProperty("Beatmapper","Video reconstruction").withProperty("AudioFile","audio.m4a").withProperty("IllustrationFile","cover.jpg");','const b=m.timing(0,60,4);','const flowComp=1.66/(Number(m.env("user.flow_speed"))||1.66);']
    for e in sorted(events,key=lambda e:e['time']):
        path=e.get('path',[]);rotation=90.
        if len(path)>2:
            subset=path[-min(8,len(path)):];a0=subset[0];a1=subset[-1];dt=a1[0]-a0[0]
            if dt>.001 and math.hypot(a1[1]-a0[1],a1[2]-a0[2])>8:
                rotation=(90+math.degrees(math.atan2(a1[1]-a0[1],a1[2]-a0[2])))%360
        angle=math.radians(90-rotation);ca=math.cos(angle);sa=math.sin(angle)
        x=(float(e['x'])/1280-.5)*1920;jy=float(e.get('judgeY',judge_y));y=(.5-jy)*1080;t=max(0,float(e['time']));end=t+max(0,float(e.get('duration',0)));flow=float(e.get('speed',950))/(120*1.66*720/1080)
        if not all(math.isfinite(v) for v in (x,jy,y,t,end,flow)):raise ValueError('音符包含无效数字')
        if int(e['type']) not in (0,1,2):raise ValueError('类型必须为 0、1、2')
        out += ['{ const l=m.line();',f'm.animation(b,0,0,0,{x:.5f},{x:.5f},0,l,0,0,false,"");',f'm.animation(b,0,0,1,{y:.5f},{y:.5f},0,l,0,0,false,"");',f'm.animation(b,0,0,4,{rotation:.5f},{rotation:.5f},0,l,0,0,false,"");',f'm.animation(b,0,0,5,{flow:.5f}*flowComp,{flow:.5f}*flowComp,0,l,0,0,false,"");','m.animation(b,0,0,8,0,0,0,l,0,0,false,"");','m.animation(b,0,0,9,0,0,0,l,0,0,false,"");',f'const q=m.note(l,b,{t:.5f},{end:.5f},{int(e["type"])},{str(bool(e.get("isFake",False))).lower()},false);']
        path=e.get('path',[])
        if len(path)>=2:
            shift=t-float(e.get('sourceTime',e['time']));xs=float(e['x'])-float(e.get('sourceX',e['x']))
            points=[]
            for a,px,py in path:
                if float(a)+shift>=t:continue
                dx=(float(px)+xs-float(e['x']))*1.5;dy=(jy*720-float(py))*1.5
                points.append((max(0,float(a)+shift),dx*ca-dy*sa,dx*sa+dy*ca))
            points.append((t,0,0))
            if len(points)>1 and points[0][0]>0:
                out.append('m.animation(b,0,0,2,0,0,1,q,0,0,false,"");')
                out.append(f'm.animation(b,{points[0][0]:.5f},{points[0][0]:.5f},2,1,1,1,q,0,0,false,"");')
            for a,z in zip(points,points[1:]):
                if z[0]<=a[0]:continue
                for key in (0,1):
                    out.append(f'm.animation(b,{a[0]:.5f},{z[0]:.5f},{key},{a[key+1]:.5f},{z[key+1]:.5f},1,q,0,0,false,"");')
        out.append('}')
    for effect in effects or []:
        samples=effect.get('samples',[])
        if not samples:continue
        out.append('{ const s=m.storyboardObject(0,"builtin.line",0);')
        out.append('m.animation(b,0,0,2,0,0,2,s,0,0,false,"");')
        start=samples[0]['time'];finish=samples[-1]['time']+.1
        out.append(f'm.animation(b,{start:.5f},{start:.5f},2,1,1,2,s,0,0,false,"");')
        out.append(f'm.animation(b,{finish:.5f},{finish:.5f},2,0,0,2,s,0,0,false,"");')
        for first,last in zip(samples,samples[1:]):
            if last['time']<=first['time']:continue
            for key,name in ((0,'x'),(1,'y'),(4,'rotation'),(10,'width')):
                out.append(f'm.animation(b,{first["time"]:.5f},{last["time"]:.5f},{key},{first[name]:.5f},{last[name]:.5f},2,s,0,0,false,"");')
        out.append('}')
    return '\n'.join(out)+'\n' 
