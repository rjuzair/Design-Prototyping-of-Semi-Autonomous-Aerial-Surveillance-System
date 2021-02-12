import cv2, math
import numpy as np
from Erosion_Dilation import applyMorphOp
from get_points import get_points

def checkDiff(img4):
    contours ,_ = cv2.findContours(img4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < 200 or area > 2000):
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img4, (x,y), (x+w,y+h), (0,0,0), cv2.FILLED)

    img_dilation = applyMorphOp(img, 0)
    img_dilation = applyMorphOp(img_dilation, 0)
    img_dilation = applyMorphOp(img_dilation, 0)
    img_dilation = applyMorphOp(img_dilation, 0)
    if np.mean(img_dilation) == 0:
        return (img_dilation, 0)
    else:
        return (img_dilation, 1)

def drawImage(img_dilation, img4, img2, orig):     
    contours ,_ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Changes Detected {0}".format(len(contours)))
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.rectangle(orig, (x-10,y-10), (x+10+w,y+10+h), (0,0,0), 3)
    return img, contours


def drawMap(Map, contours, theta, x_map, y_map, xm, ym):
    sin, cos, pi = math.sin, math.cos, math.pi
    theta = theta * pi/180
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x_r = (x - xm) * cos(theta) - (y - ym) * sin(theta) + xm
        y_r = (x - xm) * sin(theta) + (y - ym) * cos(theta) + ym
        x, y = int(x_r + x_map), int(y_r + y_map)
        Map = cv2.rectangle(Map, (x-10,y-10), (x+10+w,y+10+h), (0,0,0), 3)
    return Map  

def Draw(img4, img2, orig, Map, extracted_img, y_cord, theta):
    img_dilation, x = checkDiff(img4)
    
    if x == 0:
        img = []
        print('No Changes Detected')
        x_map = 0
        y_map = 0
        return img, Map, x_map, y_map

    elif x == 1:
        img, contours = drawImage(img_dilation, img4, img2, orig)
        try:
            xm, ym = int(img2.shape[1]/2),int(img2.shape[0]/2)
            x_map, y_map = get_points(extracted_img, orig, theta)
            y_map = y_map + y_cord
            Map = drawMap(Map, contours, theta, x_map, y_map, xm, ym)
        except:
            Map = Map
        return img, Map, x_map, y_map
