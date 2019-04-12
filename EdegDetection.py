# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 06:37:25 2019

@author: Sir
"""

import cv2
import numpy as np

#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10  #minValue
CANNY_THRESH_2 = 200  #maxValue
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.8,0.8,0.4) # In BGR format
#MASK_COLOR = (0.0,0.0,0.0) # In BGR format


#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
img = cv2.imread('C://Users//Sir//Desktop//wiener_result.jpg')
#img = cv2.imread('C://Users//Sir//Desktop//threshold.jpg')
#grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

#== Edge detection =================================================================

#MORPHOLOGICAL OPERATIONS
edges = cv2.Canny(img, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)    #[DILATION]
edges = cv2.erode(edges, None)    #Decreased white region near image boundaries [EROSION]

#== Find contours in edges, sort by area ===========================================-
contour_info = []
_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# Previously, for a previous version of cv2, this line was: 
#  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),         #TRUE_OR_FALSE
        cv2.contourArea(c),             #Calculate contour area
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

#== Create empty mask, draw filled polygon on it corresponding to largest contour ===
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

#== Smooth mask, then blur it ========================================================
mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#== Blend masked img into MASK_COLOR background =====================================
mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0                 #  for easy blending

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

cv2.imshow('img', masked)                                   # Display
cv2.waitKey()

cv2.imwrite('C:/Users/Sir/Desktop/masked.jpg', masked)           # Save