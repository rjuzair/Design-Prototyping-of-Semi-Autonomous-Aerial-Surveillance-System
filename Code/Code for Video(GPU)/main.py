import cv2, os, time, warnings
import numpy as np
from extraction_using_features import img_ex
from Drawing import Draw
from perspective_transformation import transform
from imutils import paths
from Difference import Difference

def main(threshold, Video, Map):
    s = time.time()
    images = []
    counter = 0
    pos = []
    y = []
    d = 1
    current = os.getcwd()
    warnings.filterwarnings("ignore")

    Map1 = Map
    h1, w1 = Map.shape[:2]
    img1 = cv2.cvtColor(Map, cv2.COLOR_BGR2GRAY)
        
    while True:
        ret, frame = Video.read()
        if ret:
            if counter%10 == 0:
                images.append(frame)
            counter = counter + 1
        else:
            break

    for loop in range (0, len(images) + 1, 1):
        try:
            d = loop
            start = time.time()
            orig_image = images[loop] 
            orig_gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            
            extracted_img, y_cord = img_ex(img1, orig_gray)
            transformed_img, theta = transform(orig_gray, extracted_img)

            Diff = Difference(transformed_img, orig_gray, threshold)
            Result_img, Map, x_map, y_map = Draw(Diff, transformed_img, orig_image, Map, extracted_img, y_cord, theta)
            
            if(np.shape(Result_img) != (0,)):
                filename = current + "/Result_Images/difference_img_%d.jpg"%d
                cv2.imwrite(filename, Result_img)
            
                end = time.time()
                t = end - start
                print('Time taken by Image %d:' %d, t)
                print(' ')

                '''position = x_map, y_map, theta, d
                if not os.path.isfile('points.txt'):
                   file = open("points.txt", 'w')
                else:
                    file = open("points.txt", 'a')
                file.write(str(position[0]))
                file.write(',')
                file.write(str(position[1]))
                file.write(',')
                file.write(str(position[2]))
                file.write(',')
                file.write(str(position[3]))
                file.write('\n')
                file.close()'''
                
            else:
                end = time.time()
                t = end - start
                print('Time taken by Image %d:' %d, t)
                print(' ')
                
        except:
            print('Skipped')

    filename = current + "/Result_Images/Map.jpg"
    cv2.imwrite(filename, Map1)
                
    print('Total Time:', (time.time()-s))
