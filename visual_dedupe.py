"""Collapse repeated scoreless sprite tracks using measured path overlap."""
import numpy as np


def collapse_duplicate_visual_tracks(events, min_overlap=.08,
                                    median_distance=10., close_distance=16.):
    """Remove duplicate visual detections without merging scored notes.

    A pair must share a sustained segment of its measured path. At least one
    candidate must be scoreless; two real notes are always retained even when
    their geometry happens to overlap.
    """
    paths=[]
    for event in events:
        path=np.asarray(event.get('path',[]),float)
        paths.append(path if path.ndim==2 and len(path)>=4 and path.shape[1]>=3 else None)

    def rank(i):
        event=events[i];path=paths[i]
        confidence={'high':3,'medium':2,'low':1}.get(event.get('confidence'),0)
        return (not bool(event.get('isFake')),len(path) if path is not None else 0,
                int(event.get('frames',0)),confidence)

    removed=set();pairs=[]
    for i,a in enumerate(events):
        pa=paths[i]
        if pa is None or i in removed:continue
        for j in range(i+1,len(events)):
            b=events[j];pb=paths[j]
            if pb is None or j in removed:continue
            if not (a.get('isFake') or b.get('isFake')):continue
            if a.get('type') is not None and b.get('type') is not None and int(a['type'])!=int(b['type']):continue
            lo=max(float(pa[0,0]),float(pb[0,0]));hi=min(float(pa[-1,0]),float(pb[-1,0]))
            if hi-lo<min_overlap:continue
            samples=np.unique(np.concatenate((
                pa[(pa[:,0]>=lo)&(pa[:,0]<=hi),0],
                pb[(pb[:,0]>=lo)&(pb[:,0]<=hi),0],
                np.linspace(lo,hi,max(5,int((hi-lo)*30)+1)),
            )))
            if len(samples)<4:continue
            aa=np.column_stack([np.interp(samples,pa[:,0],pa[:,k]) for k in (1,2)])
            bb=np.column_stack([np.interp(samples,pb[:,0],pb[:,k]) for k in (1,2)])
            distances=np.linalg.norm(aa-bb,axis=1)
            if np.median(distances)>median_distance or (distances<close_distance).mean()<.75:continue
            keeper,drop=(i,j) if rank(i)>=rank(j) else (j,i)
            removed.add(drop);pairs.append((keeper,drop))

    if removed:
        events[:]=[event for i,event in enumerate(events) if i not in removed]
    return {'removedDuplicateVisualTracks':len(removed),
            'removedEventIndices':sorted(removed),
            'keptDroppedPairs':pairs}
