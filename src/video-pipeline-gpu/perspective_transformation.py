from silx.opencl import sift
import numpy as np
import cv2, os, math

def transform(diff_image, map_patch):
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
        h, w = diff_image.shape[::-1]

        pad = np.zeros(map_patch.shape)
        pad[: diff_image.shape[0],: diff_image.shape[1]] = diff_image
        diff_image = pad

        sift_ocl1 = sift.SiftPlan(template=diff_image, devicetype="GPU")
        sift_ocl2 = sift.SiftPlan(template=map_patch, devicetype="GPU")

        kp1 = sift_ocl1.keypoints(diff_image)
        kp2 = sift_ocl2.keypoints(map_patch)

        sift_m = sift.MatchPlan(devicetype = "GPU")
        matches = sift_m.match(kp1, kp2)
        sa = sift.LinearAlign(diff_image, devicetype="GPU")

        transformed_img = sa.align(map_patch)
        transformed_img = transformed_img[: w, : h]

        theta_rad = abs(matches.angle[0][0]) + abs(matches.angle[0][1])
        theta = (theta_rad * 180)/math.pi
                
        return transformed_img, theta
                


    
    
