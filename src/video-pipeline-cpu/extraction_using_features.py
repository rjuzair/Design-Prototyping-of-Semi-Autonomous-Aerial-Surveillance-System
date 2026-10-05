import cv2
import numpy as np
import numpy

def img_ex(img1, img2):

        try:
                w1, h1 = img1.shape[:2]
                w2, h2 = img2.shape[:2]

                img3 = cv2.resize(img1, dsize=(int(h1/2), int(w1/2)), interpolation=cv2.INTER_CUBIC)
                img4 = cv2.resize(img2, dsize=(int(h2), int(w2)), interpolation=cv2.INTER_CUBIC)

                w3, h3 = img3.shape[:2]
                w4, h4 = img4.shape[:2]

                img3 = cv2.UMat(img3)
                img4 = cv2.UMat(img4)

                sift = cv2.xfeatures2d.SIFT_create(400)

                kp1, des1 = sift.detectAndCompute(img3, None)
                kp2, des2 = sift.detectAndCompute(img4, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                                                  ]).reshape(-1, 1, 2)

                pt1 = 2*(src_pts[0,0])

                y1 = int(pt1[1] - 3000)
                y2 = int(pt1[1] + 3000)

                if(y1 < 0):
                    y2 = y2 + abs(y1)
                    y1 = 0
                if(y2 > w1):
                    y = y2 - w1
                    y2 = w1
                    y1 = y1 - y

                img1 = img1[y1:y2,0:w1]

                if(y1 < 10):
                    return img1, y1
                elif(y1 % 10 <= 5):
                    while(y1 % 10 != 0):
                        y1 = y1 - 1
                elif(y1 % 10 > 5):
                    while(y1 % 10 != 0):
                        y1 = y1 + 1

        except:
                w1, h1 = img1.shape[:2]
                w2, h2 = img2.shape[:2]

                img3 = cv2.resize(img1, dsize=(int(h1/2), int(w1/2)), interpolation=cv2.INTER_CUBIC)
                img4 = cv2.resize(img2, dsize=(int(h2/2), int(w2/2)), interpolation=cv2.INTER_CUBIC)

                w3, h3 = img3.shape[:2]
                w4, h4 = img4.shape[:2]

                img3 = cv2.UMat(img3)
                img4 = cv2.UMat(img4)

                surf = cv2.xfeatures2d.SURF_create(3000)

                kp1, des1 = surf.detectAndCompute(img3, None)
                kp2, des2 = surf.detectAndCompute(img4, None)

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                good = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good.append(m)

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                                                  ]).reshape(-1, 1, 2)

                pt1 = 2*(src_pts[0,0])

                y1 = int(pt1[1] - 3000)
                y2 = int(pt1[1] + 3000)

                if(y1 < 0):
                    y2 = y2 + abs(y1)
                    y1 = 0
                if(y2 > w1):
                    y = y2 - w1
                    y2 = w1
                    y1 = y1 - y

                img1 = img1[y1:y2,0:w1]

                if(y1 < 10):
                    return img1, y1
                elif(y1 % 10 <= 5):
                    while(y1 % 10 != 0):
                        y1 = y1 - 1
                elif(y1 % 10 > 5):
                    while(y1 % 10 != 0):
                        y1 = y1 + 1
        print('Patch: ',y1)
                
        return img1, y1


