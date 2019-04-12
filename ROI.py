# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 05:07:51 2019

@author: Sir
"""

import cv2
import numpy as np
from PIL import Image


if __name__ == '__main__' :
     # Read image
    im = cv2.imread('C://Users//Sir//Desktop//wiener_result.jpg')
     
    # Select ROI
 #   fromCenter = False
#    r = cv2.selectROI(im, fromCenter)
    r = cv2.selectROI("Image", im, False, False)

     
    # Crop image
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
    cv2.imwrite('C://Users//Sir//Desktop//ROI.jpg',imCrop)
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)

    
"""
image = cv2.imread('C://Users//Sir//Desktop//wiener_result.jpg')
cv2.imshow("original", image)
cropped = image[70:170, 440:540]
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
"""

"""
img = Image.open('C://Users//Sir//Desktop//wiener_result1.jpg')
#area = (10,140,480,440)
area = (200,190,480,440)
cropped_image = img.crop(area)
cropped_image.show()
cv2.imshow("cropped", cropped_image)
"""

"""
img = cv2.imread('C://Users//Sir//Desktop//wiener_result.jpg')
cv2.imshow("Wave",img)
cv2.waitKey(0)
#we readed the image

mask = np.zeros(img.shape[:2],dtype="uint8")
(cx,cy) = (img.shape[1]/2,img.shape[0]/2)
cv2.rectangle(mask,(cx-75,cy-75),(cx+75,cy+75),255,-1)
cv2.imshow("Mask",mask)
cv2.waitKey(0) 

masked = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow("Masked",masked)
cv2.waitKey(0)
"""