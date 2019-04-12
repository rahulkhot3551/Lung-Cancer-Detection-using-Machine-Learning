# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:48:03 2019

@author: root
"""

import cv2

img=cv2.imread('C:/Users/Sir/Desktop/demo.jpg')
#img=cv2.imread('C:/Users/Sir/Desktop/abc.jpg')
cv2.imshow('Original',img)

final = cv2.medianBlur(img, 5)
cv2.imshow('Final',final)
cv2.imwrite('C:/Users/Sir/Desktop/final.jpg',final )

cv2.waitKey(0)
cv2.destroyAllWindows()