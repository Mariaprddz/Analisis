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


#aplicamos sobel a una imagen con ruido
img_sobel_g=filters.sobel(img_gauss)

#realizamos el padding con img sobel
img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')

#matriz para almacenar 
values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))
#aplicamos algoritmo anisotropico
cont=0
row,col = img_sobel_g_pad.shape

img_gauss_pad=np.pad(img_gauss, 1, mode='reflect') #padding de la original
while cont<20:
    cont=cont+1
    
    for i in range(1,img_sobel_g_pad.shape[0]-1):
        for j in range(1,img_sobel_g_pad.shape[1]-1):

            parche_sobel = np.array([[img_sobel_g_pad[i-1,j-1],img_sobel_g_pad[i-1,j],img_sobel_g_pad[i-1,j+1]], 
                                   [img_sobel_g_pad[i,j-1],img_sobel_g_pad[i,j],img_sobel_g_pad[i,j+1]], 
                                    [img_sobel_g_pad[i+1,j-1],img_sobel_g_pad[i+1,j],img_sobel_g_pad[i+1,j+1]]])
            
            
            
            gradiente= np.sum(parche_sobel)
                
            if gradiente<10:
                parche_gauss = np.array([[img_gauss_pad[i-1,j-1],img_gauss_pad[i-1,j],img_gauss_pad[i-1,j+1]], 
                                   [img_gauss_pad[i,j-1],img_gauss_pad[i,j],img_gauss_pad[i,j+1]], 
                                    [img_gauss_pad[i+1,j-1],img_gauss_pad[i+1,j],img_gauss_pad[i+1,j+1]]])
                mean=np.mean(parche_gauss)
                values[i-1, j-1]=mean
            else:
                values[i-1, j-1]=img_gauss[i-1, j-1]

#plot de la img gauss vs img anisotropica

fig = plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.imshow(img_gauss, cmap=plt.cm.gray)
plt.title('Original image'), plt.axis('off')
plt.subplot(122)
plt.imshow(values, cmap=plt.cm.gray)
plt.title('Filtro Anisotrópico'), plt.axis('off')         
            

                