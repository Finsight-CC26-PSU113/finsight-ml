import psutil, os, sys, time
sys.path.insert(0, '/mnt/d/OCR FinSight')

def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Baseline:              {mem():.0f} MB")

import tensorflow as tf
print(f"After TF import:       {mem():.0f} MB")

from src.online_learning import OnlineLearningModel
ol = OnlineLearningModel()
print(f"After BiLSTM load:     {mem():.0f} MB")

import easyocr
reader = easyocr.Reader(['en','id'], gpu=False, verbose=False)
print(f"After EasyOCR (CPU):   {mem():.0f} MB")

import numpy as np
dummy = (255 * np.random.rand(800, 600, 3)).astype('uint8')
t0 = time.time()
result = reader.readtext(dummy)
elapsed = time.time() - t0
print(f"After inference:       {mem():.0f} MB")
print(f"Inference time (CPU):  {elapsed:.1f}s")
print(f"")
print(f"TOTAL RAM:             {mem():.0f} MB")
