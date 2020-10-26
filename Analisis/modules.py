# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:25:34 2020

@author: nakag
"""

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import scipy.ndimage.filters as filters

def add_gnoise(n_type,image,sigma):
    if n_type=='gauss':
        gaussian_noise=np.random.normal(loc=0.0, scale=sigma,size=np.shape(image))
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
    
#NON LOCAL MEANS ALGORITHM
# Non Local Mean filter



#padding 


#plt.imshow(img)
def mean_filter(img):
    size_filter = 3
    # the filter is divided by size_filter^2 for normalization
    mean_filter = np.ones((size_filter,size_filter))/np.power(size_filter,2)
    # performing convolution
    img_meanfiltered = filters.convolve(img, mean_filter,mode='reflect')
    return img_meanfiltered


def median_filter(img, size):
    img_filtered = filters.median_filter(img,size = size,mode='reflect')
    return img_filtered

def gaussian_filter(img,sigma):
    # performing convolution
    img_gaussianfiltered = filters.gaussian_filter(img, sigma=sigma,mode='reflect')
    return img_gaussianfiltered


    
#NLM: filtrar la imagen mediante el promedio ponderado de 
#los diferentes píxeles de la imagen en función de su similitud
#con el píxel original.

def nlm(img, h_square):
#padding 

    img = np.pad(img,1, mode='reflect') #NO SE SI EL PADDING ESTA BIEN

    
    
    row,col = img.shape
    
    
    matriz_pesos = np.zeros(shape=(img.shape[0],img.shape[1])) #aqui almacenamos los pesos
    
    matriz_imagen = np.ones(shape=(img.shape[0],img.shape[1]))

    for i in range(1,img.shape[0]-1):
        #print(i)
        for j in range(1,img.shape[1]-1):
            #print(j)
            
            #parche 3x
            
            matriz1 = np.array([[img[i-1,j-1],img[i-1,j],img[i-1,j+1]], 
                                [img[i,j-1],img[i,j],img[i,j+1]], 
                                [img[i+1,j-1],img[i+1,j],img[i+1,j+1]]])
            
            #print(matriz1)
            for x in range(0,row-2):
                for y in range(0,col-2):
                    
                        matriz2 = np.array([[img[x-1,y-1],img[x-1,y],img[x-1,y+1]], 
                                [img[x,y-1],img[x,j],img[x,y+1]], 
                                [img[x+1,y-1],img[x+1,y],img[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    
                        weights_ij = (np.exp(-distance/h_square))
                    
                        matriz_pesos[x,y] = weights_ij 
                    
                        #print(weights_ij)
                    
                    #print(matriz2)
                    
                    #print(distance)
            
            
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            #matriz_ponderada = np.squeeze(np.asarray(matriz_ponderada))
            #img_gauss = np.squeeze(np.asarray(img_gauss))
            
            
            matriz_imagen[i-1,j-1] = np.dot(img,np.transpose(matriz_ponderada),out=None)
            #matriz_imagen[i-1,j-1] = np.transpose(img_gauss)* matriz_ponderada
            
            #img_gauss[i-1,j-1] = np.sum(matriz_pesos)
            
    return matriz_imagen
    #img_nlm = np.dot(matriz_pesos,img_gauss) # ESTO DA ERROR 
        
        
    
    
    
    #img_nlm = NLM(img)




                



















