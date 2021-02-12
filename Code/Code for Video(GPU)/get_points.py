import cv2, math, imutils
import numpy as np
from silx.opencl import sift

def get_points(extracted_img, difference_img, theta):

        difference_img = cv2.cvtColor(difference_img, cv2.COLOR_BGR2GRAY)
        difference_img = imutils.rotate_bound(difference_img, theta)
        w1, h1 = extracted_img.shape[::-1]
        w2, h2 = difference_img.shape[::-1]

        extracted_img = cv2.resize(extracted_img, dsize=(int(w1/10), int(h1/10)), interpolation=cv2.INTER_CUBIC)
        difference_img = cv2.resize(difference_img, dsize=(int(w2/10), int(h2/10)), interpolation=cv2.INTER_CUBIC)

        res = cv2.matchTemplate(extracted_img, difference_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        x = int(10 * max_loc[0])
        y = int(10 * max_loc[1])
        
        return x, y


