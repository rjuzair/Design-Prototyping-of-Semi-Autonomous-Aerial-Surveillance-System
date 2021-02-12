import cv2
import numpy as np
from Erosion_Dilation import applyMorphOp
from crop import crop
from Temp_Matching import temp_match
from rotate_image import rotate

def checkDiff(img4):
    contours ,_ = cv2.findContours(img4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < 1000 or area > 10000):
            x,y,w,h = cv2.boundingRect(cnt)
            img = cv2.rectangle(img4, (x,y), (x+w,y+h), (0,0,0), cv2.FILLED)
    #cv2.imwrite('sconnected.jpg', img)

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
        img5 = cv2.rectangle(img2, (x-10,y-10), (x+10+w,y+10+h), (0,255,0), 3)
    return (img, img5, contours)

def drawMap(Map, angle, transformed_img, contours):
    grayMap = cv2.cvtColor(Map, cv2.COLOR_BGR2GRAY)
    rotated = rotate(grayMap, angle)
    x3, y3 = temp_match(rotated, transformed_img)
    Map1 = rotate(Map, angle)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        Map1 = cv2.rectangle(Map1, (x3+x-10,y3+y-10), (x3+x+10+w,y3+y+10+h), (0,0,0), 3)
    Map1 = rotate(Map1, -angle)
    Map1 = crop(grayMap, Map1)
    return Map1

def Draw(img4, img2, orig, Map, angle):
    img_dilation, x = checkDiff(img4)

    if x == 0:
        img = []
        img5 = []
        print('No Changes Detected')
        return img, img5, Map

    elif x == 1:
        img, img5, contours = drawImage(img_dilation, img4, img2, orig)
        Map = drawMap(Map, angle, img2, contours)
        return img, img5, Map
