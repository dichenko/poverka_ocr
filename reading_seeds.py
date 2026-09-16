"""Paddle-only worker: return unlabelled image-derived counter row proposals."""
import argparse
import json
import sys
from pathlib import Path
from contextlib import redirect_stdout
import cv2
import numpy as np
from main import create_engine,load_image,run_ocr,normalize_ocr_result
from counter_reader import locate_counter_rows

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with redirect_stdout(sys.stderr):
        engine=create_engine()
        from paddleocr import TextRecognition
        rec=TextRecognition(model_name='en_PP-OCRv5_mobile_rec',device='cpu',enable_mkldnn=False,cpu_threads=1)
        result={}
        for f in sorted(args.input.iterdir()):
            if f.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:continue
            try:
                image=load_image(f)
                result[f.name]=locate_counter_rows(image,normalize_ocr_result(run_ocr(engine,image)),rec)
            except Exception as exc:
                result[f.name]={'error':str(exc)}
            print(f.name,'row proposals:',len(result[f.name]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
