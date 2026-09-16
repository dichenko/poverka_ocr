"""Grayscale geometry and ink isolation for mechanical counter windows."""
import cv2
import numpy as np

def rotate(image, angle):
    h,w=image.shape[:2]
    m=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    nw=int(h*abs(m[0,1])+w*abs(m[0,0])); nh=int(h*abs(m[0,0])+w*abs(m[0,1]))
    m[:,2]+=[(nw-w)/2,(nh-h)/2]
    return cv2.warpAffine(image,m,(nw,nh),borderValue=(255,255,255)),m

def angles(image):
    scale=800/max(image.shape[:2]); small=cv2.resize(image,None,fx=scale,fy=scale)
    edge=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),60,160)
    lines=cv2.HoughLinesP(edge,1,np.pi/180,40,minLineLength=65,maxLineGap=8)
    buckets={}
    for x1,y1,x2,y2 in ([] if lines is None else lines[:,0]):
        a=np.degrees(np.arctan2(y2-y1,x2-x1)); a=(a+90)%180-90
        a=(a+45)%90-45
        k=round(a/3)*3; buckets[k]=buckets.get(k,0)+np.hypot(x2-x1,y2-y1)
    top=sorted(buckets,key=buckets.get,reverse=True)[:3]
    return list(dict.fromkeys([0]+top+[a-90 if a>0 else a+90 for a in top]))

def proposals(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    found=[]
    for block in (31,61,101):
        mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,9)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        boxes=[(x,y,bw,bh) for x,y,bw,bh,area in stats[1:] if h*.018<bh<h*.22 and .08< bw/bh <1.15 and area>bh*1.5]
        for x,y,bw,bh in boxes:
            group=[b for b in boxes if .6< b[3]/bh <1.5 and abs((b[1]+b[3]/2)-(y+bh/2))<bh*.35]
            group=sorted(group)
            chains=[]; chain=[]
            for b in group:
                if chain and b[0]-(chain[-1][0]+chain[-1][2])>bh*.85:
                    chains.append(chain); chain=[]
                chain.append(b)
            chains.append(chain)
            for chain in chains:
                if len(chain)<4: continue
                x0=min(b[0] for b in chain); y0=min(b[1] for b in chain)
                x1=max(b[0]+b[2] for b in chain); y1=max(b[1]+b[3] for b in chain)
                if 2.6<(x1-x0)/(y1-y0)<12:
                    box=(x0,y0,x1,y1)
                    if not any(abs(x0-a)<10 and abs(y0-b)<10 and abs(x1-c)<10 and abs(y1-d)<10 for a,b,c,d in found): found.append(box)
    return sorted(found,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)[:12]

def grids(image,box,count=8):
    x0,y0,x1,y1=box;h=y1-y0;w=x1-x0
    if count < 1 or h < 4 or w < 4 or y0 < 0 or y1 > image.shape[0]:
        return []
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)
    left=max(0,int(x0-h*3));right=min(gray.shape[1],int(x1+h*3))
    band=gray[y0:y1,left:right]
    # Repeated vertical window boundaries have strong gradients spanning height.
    grad=np.abs(cv2.Sobel(band,cv2.CV_32F,1,0,ksize=3))
    profile=grad[int(h*.15):int(h*.85)].mean(axis=0)
    profile=cv2.GaussianBlur(profile[None],(0,0),max(1,h*.02))[0]
    best=[]
    for pitch in range(max(12,round(h*.42),round(w/9)),min(round(h*.78),round(w/3))+1):
        for start in range(max(0,round(x0-left-pitch*3)),min(len(profile)-count*pitch,round(x0-left+pitch*.6)),2):
            chunks=np.array([profile[start+i*pitch:start+(i+1)*pitch] for i in range(count)])
            # Correlation of window-edge profiles across cells; tolerate rollover.
            norms=(chunks-chunks.mean(axis=1,keepdims=True))/(chunks.std(axis=1,keepdims=True)+1)
            consistency=float(np.mean(np.sort(norms.mean(axis=0))[-max(2,pitch//8):]))
            coverage=max(0,min(start+count*pitch,x1-left)-max(start,x0-left))/w
            score=consistency*coverage
            best.append((score,start+left,pitch,y0,y1))
    chosen=[]
    for g in sorted(best,reverse=True):
        if not any(abs(g[1]-q[1])<g[2]*.3 and abs(g[2]-q[2])<g[2]*.1 for q in chosen):chosen.append(g)
        if len(chosen)==4:break
    return chosen

def clean(crop,trim=.12,threshold='otsu'):
    if crop.size == 0 or min(crop.shape[:2]) < 4:
        return np.full((96,64),255,np.uint8)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    g=gray[round(h*.05):round(h*.95),round(w*trim):round(w*(1-trim))]
    if threshold=='otsu':_,mask=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:mask=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)
    n,l,stats,centres=cv2.connectedComponentsWithStats(mask)
    choices=[]
    for k in range(1,n):
        x,y,cw,ch,area=stats[k];cx,cy=centres[k]
        if ch>g.shape[0]*.35 and .12*g.shape[1]<cx<.88*g.shape[1] and area<mask.size*.65 and cw<g.shape[1]*.92:
            choices.append((area,k))
    if not choices:return np.full((96,64),255,np.uint8)
    _,k=max(choices);x,y,cw,ch,area=stats[k]
    glyph=np.where(l[y:y+ch,x:x+cw]==k,0,255).astype('uint8')
    # The OCR model sees isolated ink, with no slot border or neighbouring roll.
    tw=min(48,max(12,round(cw*76/ch)));glyph=cv2.resize(glyph,(tw,76))
    canvas=np.full((96,64),255,np.uint8);canvas[10:86,(64-tw)//2:(64-tw)//2+tw]=glyph
    return canvas

