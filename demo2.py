import cv2
import png
from skimage.feature import greycomatrix,greycoprops
import skimage


import numpy as np
import matplotlib.pyplot as plt



import pywt
import pywt.data

from PIL import Image


# Load image
#original=cv2.imread('C:/Users/Sir/Desktop/demo.jpg')
#original = np.asarray(Image.open('C:/Users/Sir/Desktop/masked.jpg'))
original = np.asarray(Image.open('C:/Users/Sir/Desktop/threshold1.jpg'))
#print(type(original))
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'db1')
LL, (LH, HL, HH) = coeffs2

'''fig = plt.figure(figsize=(8, 5))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
'''
#print(LL)
#print(type(LL))

#img = Image.fromarray(LL)
#img.show()

#fig.tight_layout()
#plt.show()



titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(LL, 'db1')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(20, 5))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


###################### LL BAND OF SEGMENTATION DWT2 SAVED ####################
transform = np.array(LL).astype(np.uint8)
img = Image.fromarray(transform, 'L')     #Image Saved after LL band Transformation
img.save('C:/Users/Sir/Desktop/demo2.png')



