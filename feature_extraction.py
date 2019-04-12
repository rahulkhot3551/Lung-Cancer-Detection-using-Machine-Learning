# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:11:16 2019

@author: Sir
"""

import numpy as np
from skimage.feature import greycomatrix,greycoprops
from skimage.morphology import disk
import skimage
import cv2
import matplotlib.pyplot as plt
import pywt
import pywt.data
from PIL import Image



print (skimage.__version__)

#img = cv2.imread('C:/Users/Sir/Desktop/demo2.png')
img = cv2.imread('C:/Users/Sir/Desktop/image101.png')

print(type(img))
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #CONVERTING THE IMAGE TO GRAY
GLCM = greycomatrix(im,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=256,symmetric=False,normed=True)
#print("The corresponding GLCM matrix is")
print(GLCM)


#========================= CALCULATING FEATURES FROM THE RESPECTIVE MATRIX ===================

cont = greycoprops(GLCM,'contrast')
#entr = greycoprops(GLCM,'entropy')
coor = greycoprops(GLCM,'correlation')

diss = greycoprops(GLCM,'dissimilarity')

eng = greycoprops(GLCM, 'energy')

homo = greycoprops(GLCM, 'homogeneity')

asm = greycoprops(GLCM, 'ASM')

print(np.mean(asm))


print(np.mean(homo))

entropy = skimage.measure.shannon_entropy(GLCM)
print("Entropy")
print(entropy)


#========================= DISPLAYING THE FEATURES ============================================

print("ENERGY")
print(eng)


print("ENERGY")
print(np.mean(eng))

print("CONTRAST")
print(np.mean(cont))

#print("ENTROPY")
#print(entr)

print("CORRELARION")
print(np.mean(coor))

print("DISSIMILARITY")
print(np.mean(diss))