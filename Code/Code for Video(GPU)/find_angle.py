import cv2, math, imutils
import numpy as np
from silx.opencl import sift

def angle_find(extracted_img, difference_img, Map):

    w1, h1 = extracted_img.shape[::-1]
    w2, h2 = difference_img.shape[::-1]

    extracted_resized = cv2.resize(extracted_img, dsize=(int(w1/2), int(h1/2)), interpolation=cv2.INTER_CUBIC)
    difference_resized = cv2.resize(difference_img, dsize=(int(w2/2), int(h2/2)), interpolation=cv2.INTER_CUBIC)

    Sift = cv2.xfeatures2d.SIFT_create(4000)

    kp1, des1 = Sift.detectAndCompute(extracted_resized, None)
    kp2, des2 = Sift.detectAndCompute(difference_resized, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                                            ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                                            ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        ss = M[0, 1]
        sc = M[0, 0]
        theta = math.atan2(ss, sc) * 180 / math.pi

        rotated = imutils.rotate_bound(difference_img, theta)
        
        res = cv2.matchTemplate(extracted_img, rotated, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        x = max_loc[0]
        y = max_loc[1]
        
        return x, y, theta


