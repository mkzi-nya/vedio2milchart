"""Continue a moving hold through viewport clipping using source pixels."""
import cv2
import numpy as np
from scipy.ndimage import median_filter


def extend(video,events):
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);report=[]
    for e in events:
        p=np.asarray(e.get('bodyPath',[]),float)
        if e.get('duration',0)<.8 or len(p)<8:continue
        post=p[p[:,0]>e['time']+.05]
        if len(post)<7 or np.ptp(post[:,2])<50 or post[-1,2]<630:continue
        end=e['time']+e['duration'];start=post[0,0]
        decorative=bool(e.get('isFake'))
        # Visual lifetime is measured separately from scoring. Quiet-counter
        # evidence can make a capsule decorative without ending its animation.
        scan_end=max(end,post[-1,0]+2.5) if decorative else end
        y=median_filter(post[:,2],size=3,mode='nearest')
        vx=float(np.median(np.diff(post[:,1])/np.diff(post[:,0])))
        vy=float(np.median(np.diff(y)/np.diff(post[:,0])))
        if vy<50:continue
        x=post[0,1];head=y[0];last=start-1/fps
        anchors=[q for q in e.get('anchorPath',[]) if q[0]<start-.5/fps]
        tails=[q for q in e.get('tailPath',[]) if q[0]<start-.5/fps]
        opacity=[[e['time'],1.]];observed=0;clipped=0;paths=[];widths=[]
        for f in range(round(start*fps),round(scan_end*fps)):
            tm=f/fps;dt=tm-last;pred=x+vx*dt
            cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read()
            if not ok:break
            x0=max(0,round(pred)-65);x1=min(1280,round(pred)+66)
            if x1<=x0:break
            crop=im[:,x0:x1];white=((crop[:,:,0]>55)&(crop.min(2)>28)&(crop[:,:,1]>40)).astype('uint8')
            # The score text can join the beam into a very wide component.
            # Inspect the body below the HUD, then test the upper strip alone.
            white[:112]=0
            # Select the sustained vertical column before measuring its caps.
            # Radial hit petals otherwise inflate a component's width.
            from scipy.ndimage import label
            cols=white.sum(0)>170;groups,ng=label(cols)
            choices=[]
            for k in range(1,ng+1):
                xx=np.flatnonzero(groups==k);w=len(xx)
                if not 19<=w<=60:continue
                ys=np.flatnonzero(white[:,xx].sum(1)>=w*.55)
                if len(ys)<170:continue
                yy=int(ys[0]);h=int(ys[-1]-ys[0]+1)
                cx=x0+float(np.mean(xx))
                if abs(cx-pred)>48:continue
                choices.append((abs(cx-pred),k,cx,yy,w,h))
            if not choices:
                if observed>=8 and tm-last>.2:break
                continue
            _,k,cx,top,width,height=min(choices)
            bottom=top+height
            if clipped and bottom<715 and abs(bottom-width/2-(head+vy*dt))>50:continue
            if bottom>=715:
                head=max(720.,head+vy*dt);clipped+=1
            else:
                head=bottom-width/2
            # Source beam centre remains measurable when both caps leave the
            # camera. Do not freeze it at the last visible hit particle.
            vx=np.clip((cx-x)/max(dt,.01),-400,400)*.35+vx*.65
            x=cx;last=tm
            upper=crop[4:108,max(0,round(cx-x0)-6):round(cx-x0)+7]
            upper_filled=upper.size and ((upper[:,:,0]>85)&(upper.min(2)>55)&(upper[:,:,1]>65)).mean()>.7
            tail=-60. if top<=114 and upper_filled else top+width/2
            anchors.append([tm,float(x),float(head)])
            tails.append([tm,float(x),float(tail)])
            paths.append([tm,float(x),float(head)]);widths.append([tm,float(width)])
            # Measure the fill, excluding its bright border and hit petals.
            core=crop[max(112,top+12):min(705,bottom-12),max(0,round(cx-x0)-6):round(cx-x0)+7]
            alpha=float(np.clip(np.median(core.min(2))/235,.05,1.)) if core.size else 1.
            opacity.append([tm,alpha]);observed+=1
        if observed>=8 and clipped>=3:
            if decorative:
                visual_end=last+1/fps
                e['path']=[q for q in e.get('path',[]) if q[0]<start-.5/fps]+paths
                e['bodyPath']=e['path']
                e['holdWidthPath']=[q for q in e.get('holdWidthPath',[]) if q[0]<start-.5/fps]+widths
                e['holdRotationPath']=[q for q in e.get('holdRotationPath',[]) if q[0]<start-.5/fps]+[[q[0],90.] for q in paths]
                e['sourceVisualEnd']=visual_end
                end=visual_end
            anchors.append([end,*anchors[-1][1:]])
            e['anchorPath']=anchors;e['tailPath']=tails;e['opacityPath']=opacity
            e['offscreenHoldEvidence']='source-wide-capsule-centre-and-clipped-caps-with-moving-head'
            report.append({'time':e['time'],'observedFrames':observed,'clippedFrames':clipped})
    cap.release()
    return {'continuedHolds':len(report),'holds':report}
