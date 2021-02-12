import cv2
import numpy as np
from Temp_Matching import temp_match
def crop(Map, Map1):
    gray = cv2.cvtColor(Map1, cv2.COLOR_BGR2GRAY)

    h1, w1 = Map.shape[:2]
    h2, w2 = gray.shape[:2]

    x, y = temp_match(gray, Map)
    
    Map1 = Map1[y : y + h1, x : x + w1]
    return Map1


