import cv2
import numpy as np


def temp_match(img, template):
    
    w1, h1 = img.shape[::-1]
    w2, h2 = template.shape[::-1]

    img = cv2.resize(img, dsize=(int(w1/100), int(h1/100)), interpolation=cv2.INTER_CUBIC)
    template = cv2.resize(template, dsize=(int(w2/100), int(h2/100)), interpolation=cv2.INTER_CUBIC)
    
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    chk = np.max(result)
    loc = np.where(result >= chk)
    
    for pt in zip(*loc[::-1]):
        x = 1
     
    x3 = 100 * pt[0]
    y3 = 100 * pt[1]


    return (x3, y3)
