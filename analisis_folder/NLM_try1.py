#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:35:32 2020

@author: gemaperez
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

#%%

img = nib.load('/Users/gemaperez/Imagen/bd_schizo/sub-01/anat/sub-01_T1w.nii.gz')


#affine = img.affine


#header = img.header['pixdim']


#print( img.get_data_dtype())

#plt.imshow(img)
from nilearn import plotting
#plotting.plot_img(img, title="Prueba1")

#plotting.show()

a = np.array(img.dataobj)
#print(a)

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
plt.figure()
img_gauss=add_gnoise('gauss',img_o,0.25)
plt.imshow(img_gauss)
plt.title('Gauss')
#Aplicar ruido impulsivo
plt.figure()
img_salpimienta=salpimienta('s&p',img_o,1.0)
plt.imshow(img_salpimienta)
plt.title('Salt&Pepper')

#%%

#NLM: filtrar la imagen mediante el promedio ponderado de 
#los diferentes píxeles de la imagen en función de su similitud
#con el píxel original.

img = img_gauss

#padding 

img = np.pad(img,1, mode='reflect')
#plt.imshow(img)


row, col = img.shape



for i in range(0,row-2):
    #print(i)
    for j in range(0,col-2):
        #print(j)
        
        #parche 3x
        
        matrix1 = np.array([[img[i-1,j-1],img[i-1,j],img[i-1,j+1]], 
                            [img[i,j-1],img[i,j],img[i,j+1]], 
                            [img[i+1,j-1],img[i+1,j],img[i+1,j+1]]])
        
        print(matriz1)

        
        for x in range(0,row-2):
            for y in range(0,col-2):
                
                matrix2 = np.array([[img[x-1,y-1],img[x-1,y],img[x-1,y+1]], 
                            [img[x,y-1],img[x,j],img[x,y+1]], 
                            [img[x+1,y-1],img[x+1,y],img[x+1,y+1]]])
                
                distance = np.sqrt((matrix1-matrix2)**2)
                distance = np.sum(distance)
                
                print(matriz2)
                
                print(distance)