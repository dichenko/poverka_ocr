"""Digit-only EasyOCR on deskewed, grayscale mechanical counter windows."""
from pathlib import Path
import cv2
import numpy as np
from counter_vision import angles,proposals,rotate,grids,clean
from meter_reading import crop_cell,empty_reading

def vote(observations):
    strong={t for t,c in observations if len(t)==1 and t.isdigit() and c>=.9}
    if len(strong)!=1:return 'X',None
    digit=next(iter(strong));support=[c for t,c in observations if t==digit and c>=.35]
    if len(support)<2:return 'X',None
    return digit,float(max(support))


def vote_strength(observations, digit):
    """Strength of repeated evidence for one already-confirmed digit."""
    values=sorted((confidence for text,confidence in observations
                   if text == digit and confidence >= .35),reverse=True)
    return float(sum(values[:2])) if len(values) >= 2 else 0.0


def combine_votes(preferred, alternate, preferred_observations=None, alternate_observations=None):
    """Fill a conservative ``X`` from an independently confirmed variant.

    ``vote`` has already required repeated support inside each preprocessing
    variant.  Therefore a confirmed digit from the alternate variant is more
    informative than an unknown produced solely by a strict confidence cutoff
    in the preferred variant.  Conflicting *known* values remain governed by
    the row-level preferred variant.
    """
    if preferred[0] == 'X' and alternate[0] != 'X':
        return alternate
    if (preferred[0] != 'X' and alternate[0] != 'X' and preferred[0] != alternate[0]
            and preferred_observations is not None and alternate_observations is not None):
        # The text-line preprocessing is useful for selecting the row, but a
        # local digit can still be clearer in the other thresholded version.
        # Compare repeated evidence, never one isolated high-confidence guess.
        if vote_strength(alternate_observations,alternate[0]) > vote_strength(preferred_observations,preferred[0]):
            return alternate
    return preferred


def locate_counter_rows(image, items, seed_recognizer=None):
    """Find an unlabelled row candidate using existing OCR, then geometry.

    No filename, colour mask, or prior reading participates in this operation.
    The fallback recognizer is the already-warmed Paddle digit model.
    """
    candidates=[]
    for item in items:
        polygon=np.asarray(item['polygon'],np.float32)
        width=float(np.linalg.norm(polygon[1]-polygon[0]))
        height=float(np.linalg.norm(polygon[3]-polygon[0]))
        text=item['text']
        if (sum(char.isdigit() for char in text) >= 4
                and not any(char.isalpha() for char in text)
                and height > max(image.shape[:2])*.04 and width > 2.5*height):
            candidates.append(dict(kind='polygon',polygon=polygon.tolist(),height=height,width=width))
    if candidates:
        return [max(candidates,key=lambda candidate:candidate['height'])]
    if seed_recognizer is None:
        return []
    found=[]
    for angle in angles(image):
        turned,_=rotate(image,angle)
        boxes=[box for box in proposals(turned)
               if box[3]-box[1] > max(image.shape[:2])*.04]
        if not boxes:
            continue
        crops=[cv2.copyMakeBorder(turned[y0:y1,x0:x1],8,8,8,8,cv2.BORDER_CONSTANT,
                                  value=(255,255,255))
               for x0,y0,x1,y1 in boxes]
        for box,result in zip(boxes,seed_recognizer.predict(crops,batch_size=16)):
            text=result['rec_text']
            if (sum(char.isdigit() for char in text) >= 4
                    and not any(char.isalpha() for char in text)
                    and result['rec_score'] > .45):
                x0,y0,x1,y1=box
                found.append(dict(kind='rotated',angle=float(angle),box=list(map(int,box)),
                                  area=int((x1-x0)*(y1-y0))))
    found.sort(key=lambda candidate:candidate['area'],reverse=True)
    return found[:1]

class CounterReader:
    def __init__(self,model_dir,integer_digits=5,decimal_places=3,download_models=False):
        if integer_digits < 1 or decimal_places < 1:
            raise ValueError('The configured integer and fractional lengths must be positive')
        import torch,easyocr
        torch.set_num_threads(2)
        self.reader=easyocr.Reader(['en'],gpu=False,detector=False,
            model_storage_directory=str(model_dir),user_network_directory=str(Path(model_dir)/'user'),
            download_enabled=download_models,verbose=False)
        self.integer_digits=integer_digits;self.decimal_places=decimal_places

    def read(self,image,seeds,debug=None):
        count=self.integer_digits+self.decimal_places
        image=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        evaluated=[]
        for seed in seeds:
            if seed['kind']=='polygon':
                p=np.asarray(seed['polygon'],np.float32);w=seed['width'];h=seed['height']
                ex=p.copy();ex[[1,2]]+=(p[1]-p[0])*.8
                turned=crop_cell(image,ex);box=[0,0,round(w),round(h)]
                th,tw=turned.shape[:2]
                inverse=cv2.getPerspectiveTransform(np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]],np.float32),ex)
                angle=float(np.degrees(np.arctan2(p[1,1]-p[0,1],p[1,0]-p[0,0])))
            else:
                angle=seed['angle'];turned,m=rotate(image,angle);box=seed['box']
                inverse=np.vstack([cv2.invertAffineTransform(m),[0,0,1]])
            hypotheses=grids(turned,box,count)
            if not hypotheses:continue
            hypotheses=[g for g in hypotheses if g[0]>=hypotheses[0][0]*.88]
            for geometric_score,x,pitch,y0,y1 in hypotheses:
                crops=[turned[y0:y1,x+i*pitch:x+(i+1)*pitch] for i in range(count)]
                modes={};all_obs={}
                for mode in ('adaptive','otsu'):
                    glyphs=[clean(c,trim,mode) for trim in (.03,.12,.2) for c in crops]
                    canvas=np.concatenate(glyphs,axis=1)
                    boxes=[[i*64,(i+1)*64,0,96] for i in range(len(glyphs))]
                    results=self.reader.recognize(canvas,horizontal_list=boxes,free_list=[],allowlist='0123456789',batch_size=32,detail=1)
                    obs=[[(results[t*count+i][1],float(results[t*count+i][2])) for t in range(3)] for i in range(count)]
                    modes[mode]=[vote(o) for o in obs];all_obs[mode]=obs
                selected='adaptive'
                ai=sum(d!='X' for d,c in modes['adaptive'][:self.integer_digits]);oi=sum(d!='X' for d,c in modes['otsu'][:self.integer_digits])
                if ai<self.integer_digits-1 and oi>ai:selected='otsu'
                other='otsu' if selected == 'adaptive' else 'adaptive'
                digits=[combine_votes(primary,alternate,all_obs[selected][i],all_obs[other][i])
                        for i,(primary,alternate) in enumerate(zip(modes[selected],modes[other]))]
                # The final wheel is especially prone to rollover. Accept it only
                # when independent threshold methods agree with strong evidence.
                if modes['adaptive'][-1][0]!=modes['otsu'][-1][0]:digits[-1]=('X',None)
                cells=[]
                for i,(d,c) in enumerate(digits):
                    rect=np.array([[x+i*pitch,y0],[x+(i+1)*pitch,y0],[x+(i+1)*pitch,y1],[x+i*pitch,y1]],np.float32)
                    polygon=cv2.perspectiveTransform(rect[None],inverse)[0]
                    cells.append(dict(digit=d,confidence=c,polygon=polygon.tolist(),
                        ocr_candidates=[dict(text=t,confidence=s,preprocessing=mode) for mode in all_obs for t,s in all_obs[mode][i]]))
                text=''.join(d for d,c in digits)
                integer_known=sum(d!='X' for d,c in digits[:self.integer_digits]);known=sum(d!='X' for d,c in digits)
                # Quality is a ranking signal, not a calibrated probability.
                score=integer_known*2+known+geometric_score*.1
                evaluated.append(dict(score=score,cells=cells,digits=text,deskew_angle=angle,preprocessing=selected,geometry_score=float(geometric_score)))
        if debug is not None:debug.extend(evaluated)
        if not evaluated:return empty_reading()
        best=max(evaluated,key=lambda r:r['score'])
        if sum(d!='X' for d in best['digits'])<self.integer_digits:return empty_reading()
        text=best['digits'];return dict(value=text[:-self.decimal_places]+'.'+text[-self.decimal_places:],
            status='partial' if 'X' in text else 'recognized',decimal_places=self.decimal_places,digits_count=count,
            cells=best['cells'],error=None,method='grayscale_digit_consensus',deskew_angle=best['deskew_angle'],
            preprocessing=best['preprocessing'])
