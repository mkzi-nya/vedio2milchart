"""Compact source-sampled motion into native linear Milize animations.

Use interpolation rather than per-frame expressions: measured motion contains
MPEG/detector jitter and does not justify an expensive polynomial interpreter.
"""
import numpy as np


def curves(points, tolerance=1.):
    if not points:
        return []
    clean=[]
    for q in points:
        p=[float(q[0]),float(q[1])]
        if clean and p[0]==clean[-1][0]:
            clean[-1]=p
        else:
            clean.append(p)
    if len(clean)==1:
        return [[*clean[0][:1],clean[0][0],clean[0][1],clean[0][1]]]
    p=np.asarray(clean,float); result=[]
    def fit(a,b):
        t0,y0=p[a];t1,y1=p[b];duration=t1-t0
        if b-a<3 or duration<.002:
            if b-a==1:
                result.append([round(t0,5),round(t1,5),round(y0,5),round(y1,5)]);return
        if duration<=0:
            return
        u=(p[a:b+1,0]-t0)/duration
        predicted=y0+(y1-y0)*u
        error=abs(predicted-p[a:b+1,1])
        # Short visibility transitions keep their explicit authored timing.
        discontinuity=np.flatnonzero(np.diff(p[a:b+1,0])<.001)
        if len(discontinuity) and b-a>1:
            k=a+int(discontinuity[0])+1
            if k==b:k=b-1
        elif error.max()<=tolerance:
            segment=[round(t0,5),round(t1,5),round(y0,5),round(y1,5)]
            result.append(segment);return
        else:
            k=a+int(np.argmax(error))
            k=max(a+1,min(k,b-1))
        fit(a,k);fit(k,b)
    fit(0,len(p)-1)
    return result
