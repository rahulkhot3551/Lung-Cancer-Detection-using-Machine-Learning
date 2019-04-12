import cv2
import png
from skimage.feature import greycomatrix,greycoprops

import skimage


import numpy as np
import matplotlib.pyplot as plt



import pywt
import pywt.data

from PIL import Image


######################## PERFORMING DWT LEVEL-1 ON THE IMAGE ##################################
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



################# PASSING LL BAND TO LEVEL-2 OF THE DWT TRANSFORMATION ######################


titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(LL, 'db1')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(8, 5))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


binary_transform = np.array(LL).astype(np.uint8)
img = Image.fromarray(binary_transform, 'L')     #Image Saved after LL band Transformation
img.save('C:/Users/Sir/Desktop/image101.png')


#glc = np.asarray(Image.open('C:/Users/Sir/Desktop/image101.png'))


imge = Image.fromarray(LL)    #Converted to image 
#img_out = np.copy(imge)
#print(LL)
#img.save('C:/Users/Sir/Desktop/dwt2.jpg')
#img.show()
#abc = np.array([LL],dtype=np.uint8)


#print(abc)
#plt.savefig('C:/Users/Sir/Desktop/dwt2.jpg')\

#info = np.iinfo(data.dtype)



#img = cv2.imread('C:/Users/Sir/Desktop/masked.jpg')
#img = np.asarray(imge)
#im = cv2.cvtColor(LL, cv2.COLOR_BGR2GRAY)
#GLCM = greycomatrix(LL,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=256,symmetric=False,normed=True)
#print(GLCM)


info = np.iinfo(imge.dtype)
print(info)
#data = img.astype(np.float64)/info.max
#data = 255 * data
#img = data.astype(np.uint8)
#cv2.imshow("window",img)
#image = img_as_ubyte(imge)

#GLCM = greycomatrix(glc,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=4,symmetric=False,normed=True)
#print(GLCM)