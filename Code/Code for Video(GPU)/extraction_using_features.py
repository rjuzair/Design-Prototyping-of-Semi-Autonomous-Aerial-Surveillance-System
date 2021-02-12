from silx.opencl import sift
import numpy as np
import cv2, os, math


def img_ex(image1, image2):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        image1_resized = cv2.resize(image1, dsize=(int(w1/2), int(h1/2)), interpolation=cv2.INTER_CUBIC)

        pad = np.zeros(image1_resized.shape)
        pad[: h2,: w2] = image2
        image2_resized = pad

        sift_ocl1 = sift.SiftPlan(template=image1_resized, devicetype="GPU")
        sift_ocl2 = sift.SiftPlan(template=image2_resized, devicetype="GPU")

        kp1 = sift_ocl1.keypoints(image1_resized)
        kp2 = sift_ocl2.keypoints(image2_resized)

        sift_m = sift.MatchPlan(devicetype = "GPU")
        matches = sift_m.match(kp1, kp2)

        u = 0
        v = 1
        while(v != len(matches)):
            if v == len(matches)-1:
                u = u + 1
                v = u + 1
            dx1 = (matches.x[u][0]-matches.x[v][0])**2
            dy1 = (matches.y[u][0]-matches.y[v][0])**2
            dx2 = (matches.x[u][1]-matches.x[v][1])**2
            dy2 = (matches.y[u][1]-matches.x[v][1])**2 
            d1 = math.sqrt(dx1 + dy1)
            d2 = math.sqrt(dx2 + dy2)
            v = v + 1
            if d1 < d2 + 10 and d1 > d2 - 10:
                break

        pt1 = 2*(matches.y[0][0])

        y1 = int(pt1 - 1080)
        y2 = int(pt1 + 1080)

        if(y1 < 0):
            y2 = y2 + abs(y1)
            y1 = 0
        if(y2 > h1):
            y = y2 - h1
            y2 = h1
            y1 = y1 - y
            
        image1 = image1[y1 : y2, 0 : w1]
        
        return image1, y1


