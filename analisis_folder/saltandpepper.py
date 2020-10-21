# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:21:57 2020

@author: Maria
"""

import scipy
import numpy as np
import nibabel as nib
import matplotlib 
import matplotlib.pyplot as plt 
from skimage import io
from skimage import filters
from nilearn import datasets
from skimage.color import rgb2gray

img = nib.load(r'\Users\Maria\Desktop\data\sub-01\anat\sub-01_T1w.nii.gz')

print (img)

affine = img.affine
print(affine) 

header = img.header['pixdim']
print(header)

print( img.get_data_dtype())

#plt.imshow(img)
from nilearn import plotting
#plotting.plot_img(img, title="Prueba1")

#plotting.show()

a = np.array(img.dataobj)
print(a)

#plt.imshow(a[:,:,128],cmap=plt.cm.gray)

img_gray= a[:,:,128]


#normalizo la imagen
img_o=img_gray
img_o=img_o/np.max(img_o) 


#añadir ruido
def add_gnoise(n_type,image,sigma):
    if n_type=='gauss':
        gaussian_noise=np.random.normal(loc=0.0, scale=sigma,size=np.shape(img_o))
        noisy = image + gaussian_noise
        return noisy

def salpimienta(n_type,image,intensity):
    if n_type=='s&p' :
        cant = intensity
        ruido_output= np.copy(image)

        #ruido de sal
        num_salt = np.ceil(cant * image.size * 0.5)
        pos = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        ruido_output[pos]=1
        #ruido pimienta
        num_pepper= np.ceil(cant * image.size * 0.5)
        pos = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        ruido_output[pos]=0
        return ruido_output

#Aplicar ruido gaussiano
img_gauss=add_gnoise('gauss',img_o,0.25)
plt.imshow(img_gauss)
#Aplicar ruido impulsivo
img_salpimienta=salpimienta('s&p',img_o,1.0)
plt.imshow(img_salpimienta)

