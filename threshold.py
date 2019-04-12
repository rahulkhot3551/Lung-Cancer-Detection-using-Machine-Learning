# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:03:25 2019

@author: Sir
"""

import cv2
import numpy as np


#= Thresholding - The idea behind thresholding is to split the image into foreground and background
#= Pixel values less than or equal to threshold -> Background
#= Pixel values greater than threshold -> Foreground

#====================THRESHOLDING THE MASKED IMAGE =========================
img = cv2.imread('C://Users//Sir//Desktop//masked.jpg')
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#================== ALGORITHM USED -> ADAPTIVE THRESHOLD ===================
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.imwrite('C:/Users/Sir/Desktop/threshold1.jpg',th )
cv2.waitKey(0)
cv2.destroyAllWindows()