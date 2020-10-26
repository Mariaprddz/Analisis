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
from skimage.transform import resize 

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

img_gray= a[:,:,128]  #que nuestra función tenga un parámetro de entrada en NLM


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
plt.imshow(img_gauss, cmap=plt.cm.gray)
plt.title('Gauss')

#Aplicar ruido impulsivo
plt.figure()
img_salpimienta=salpimienta('s&p',img_o,0.25)
plt.imshow(img_salpimienta, cmap=plt.cm.gray)
plt.title('Salt&Pepper')


#%%

#filtro de mediana, media y gaussiano 



#%%

#NLM: filtrar la imagen mediante el promedio ponderado de 
#los diferentes píxeles de la imagen en función de su similitud
#con el píxel original.


#from numba import jit




#img_1 = img_gauss #lo de gauss 

#aplicamos un resize

img_gauss = resize(img_gauss, (img_gauss.shape[0] // 6, img_gauss.shape[1] // 6),
                       anti_aliasing=True)



#parameters

h_square = 1 

#padding 

#img = np.pad(img,1, mode='reflect')
#plt.imshow(img) 


row,col = img_gauss.shape

#matriz_ceros

matriz_pesos = np.zeros(shape=(img_gauss.shape[0],img_gauss.shape[1])) #aqui almacenamos los pesos

#@jit(nopython=True)


for i in range(0,img_gauss.shape[0]-2):
    #print(i)
    for j in range(0,img_gauss.shape[1]-2):
        #print(j)
        
        #parche 3x
        
        matriz1 = np.array([[img_gauss[i-1,j-1],img_gauss[i-1,j],img_gauss[i-1,j+1]], 
                            [img_gauss[i,j-1],img_gauss[i,j],img_gauss[i,j+1]], 
                            [img_gauss[i+1,j-1],img_gauss[i+1,j],img_gauss[i+1,j+1]]])
        
        #print(matriz1)
        for x in range(0,row-2):
            for y in range(0,col-2):
                
                    matriz2 = np.array([[img_gauss[x-1,y-1],img_gauss[x-1,y],img_gauss[x-1,y+1]], 
                            [img_gauss[x,y-1],img_gauss[x,j],img_gauss[x,y+1]], 
                            [img_gauss[x+1,y-1],img_gauss[x+1,y],img_gauss[x+1,y+1]]])
                
                    distance = np.sqrt((matriz1-matriz2)**2)
                    distance = np.sum(distance)
                
                    weights_ij = (np.exp(-distance/h_square))
                
                    matriz_pesos[x,y] = weights_ij 
                
                    #print(weights_ij)
                
                #print(matriz2)
                
                #print(distance)
        

img_nlm = np.dot(matriz_pesos,img_gauss) # ESTO DA ERROR 
    
    



#img_nlm = NLM(img)


plt.figure()
plt.imshow(img_gauss,cmap=plt.cm.gray)
plt.title('Con ruido gauss, antes')

plt.figure()
plt.title('Aplicado NLM, después')
plt.imshow(img_nlm, cmap=plt.cm.gray)
                
                
                
                
                
                
                
                
                
                