import cv2
from extraction_using_features import img_ex
from Drawing import Draw
from perspective_transformation import transform
from Temp_Matching import temp_match
from rotate_image import rotate
from imutils import paths
from crop import crop
from Difference import Difference
import numpy as np
import numpy
import matplotlib as plt
import argparse
import time

def main(imagePaths, Map):
        s = time.time()
        images = []
        counter = 0
        d = 1
        for imagePath in imagePaths:
                image = cv2.imread(imagePath)
                images.append(image)
                counter = counter + 1

        Map1 = Map
        img1 = cv2.cvtColor(Map, cv2.COLOR_BGR2GRAY)
        
        h, w = Map.shape[:2]

        for loop in range (0, counter, 1):
                try:
                        start = time.time()
                        image2 = images[loop]
                        img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

                        extracted_img, y = img_ex(img1, img2, 0)

                        transformed_img, angle = transform(img2, extracted_img, y)
                        filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\extracted&transformed_img_%d.jpg"%d
                        cv2.imwrite(filename, transformed_img)

                        Diff = Difference(transformed_img, img2)
                        Resultant_img, Resultant_img1, Map1 = Draw(Diff, transformed_img, image2, Map1, angle)

                        if(np.shape(Resultant_img) != (0,) or np.shape(Resultant_img1) != (0,)):
                            filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\difference_img_%d.jpg"%d
                            cv2.imwrite(filename, Resultant_img)
                            filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\difference_orig_img_%d.jpg"%d
                            cv2.imwrite(filename, Resultant_img1)
                            cv2.imwrite("C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\Difference_on_Map.jpg", Map1)
                            end = time.time()
                            t = end - start
                            print('Time taken by Image %d:' %d, t)
                            d = d + 1
                        else:
                            end = time.time()
                            t = end - start
                            print('Time taken by Image %d:' %d, t)
                            d = d + 1
                except:
                        try:
                                image2 = images[loop]
                                img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

                                extracted_img, y = img_ex(img1, img2, 1)

                                transformed_img, angle = transform(img2, extracted_img, y)
                                filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\extracted&transformed_img_%d.jpg"%d
                                cv2.imwrite(filename, transformed_img)

                                Diff = Difference(transformed_img, img2)
                                Resultant_img, Resultant_img1, Map1 = Draw(Diff, transformed_img, image2, Map1, angle)

                                if(np.shape(Resultant_img) != (0,) or np.shape(Resultant_img1) != (0,)):
                                    filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\difference_img_%d.jpg"%d
                                    cv2.imwrite(filename, Resultant_img)
                                    filename = "C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\difference_orig_img_%d.jpg"%d
                                    cv2.imwrite(filename, Resultant_img1)
                                    cv2.imwrite("C:\\Users\\ubaid\\AppData\\Local\\Programs\\Python\\Python37\\CodesandImages\\FYPFinalCode\\Result_Images\\Difference_on_Map.jpg", Map1)
                                    end = time.time()
                                    t = end - start
                                    print('Time taken by Image %d:' %d, t)
                                    d = d + 1
                                else:
                                    end = time.time()
                                    t = end - start
                                    print('Time taken by Image %d:' %d, t)
                                    d = d + 1
                        except:
                                print('Skipped')
                        
        print('Total Time:', (time.time()-s))
        return
