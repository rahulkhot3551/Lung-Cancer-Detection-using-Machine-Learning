# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:11:16 2019

@author: Sir
"""

import numpy as np
from skimage.feature import greycomatrix,greycoprops
#from skimage.filters.rank import entropy
from skimage.morphology import disk
import skimage
import cv2




import matplotlib.pyplot as plt

import pywt
import pywt.data

from PIL import Image



print (skimage.__version__)

#img = cv2.imread('C:/Users/Sir/Desktop/masked.jpg')
img = cv2.imread('C:/Users/Sir/Desktop/demo2.png')
print(type(img))
#img = np.asarray(Image.open('C:/Users/Sir/Desktop/masked.jpg'))
#img = np.asarray(Image.open('C:/Users/Sir/Desktop/demo2.png'))
print(type(img))
im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
GLCM = greycomatrix(im,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=256,symmetric=False,normed=True)
#print(GLCM)
cont = greycoprops(GLCM,'contrast')
#entr = greycoprops(GLCM,'entropy')
coor = greycoprops(GLCM,'correlation')

diss = greycoprops(GLCM,'dissimilarity')

eng = greycoprops(GLCM, 'energy')




entropy = skimage.measure.shannon_entropy(GLCM)
print(entropy)

print("ENERGY")
print(eng)

print("CONTRAST")
print(cont)

#print("ENTROPY")
#print(entr)

print("CORRELARION")
print(coor)

#print("ENERGY")
#print(ener)

#print("VARIANCE")
#print(var)

print("DISSIMILARITY")
print(diss)