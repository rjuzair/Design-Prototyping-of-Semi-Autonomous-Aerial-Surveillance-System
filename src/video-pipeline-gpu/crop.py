import cv2
import numpy as np

def crop(Map, w1, h1):
    h2, w2 = gray.shape[:2]
    w = int(w1/2)
    h = int(h1/2)
    Map = Map[h : h1 - h, w : w1 - w]
    return Map
