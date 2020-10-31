# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:25:34 2020

@author: nakag
"""

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import scipy.ndimage.filters as filters
from numba import jit

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

def nlm(img_ori, h_square):
#padding 

    img_pad = np.pad(img_ori,1, mode='reflect') #NO SE SI EL PADDING ESTA BIEN

    
    
    row,col = img_pad.shape
    
    
    matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
    
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))

    for i in range(1,img_pad.shape[0]-1):
        #print(i)
        for j in range(1,img_pad.shape[1]-1):
            #print(j)
            
            #parche 3x
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            #print(matriz1)
            for x in range(1,img_pad.shape[0]-1):
                for y in range(1,img_pad.shape[1]-1):
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,j],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    
                        weights_ij = (np.exp(-distance/h_square))
                    
                        matriz_pesos[x-1,y-1] = weights_ij 
                    
            
            
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen

def nlm_samepatch(img_ori, h_square):
#padding 

    img_pad = np.pad(img_ori,1, mode='reflect') 

    row,col = img_pad.shape
    
    
    matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
    
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))

    for i in range(1,img_pad.shape[0]-1):
        #print(i)
        for j in range(1,img_pad.shape[1]-1):
            #print(j)
            
            #parche 3x
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            #print(matriz1)
            for x in range(1,img_pad.shape[0]-1):
                for y in range(1,img_pad.shape[1]-1):
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                        
                        weights_ij = (np.exp(-distance/h_square))

                        matriz_pesos[x-1,y-1] = weights_ij 
                    
            matriz_pesos[i-1,j-1] = 0 #hago que nuestro pix sea el de menor valor, para evitas 
                        
            matriz_pesos[i-1,j-1] = np.max(matriz_pesos)
                    
            
            
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen

def nlm_cpp(img_ori, h_square, D_0, alpha):
#padding 

    img_pad = np.pad(img_ori,1, mode='reflect') 

    
    
    row,col = img_pad.shape
    
    
    matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
    
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))
    
    matriz_nu = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))

    for i in range(1,img_pad.shape[0]-1):
        #print(i)
        for j in range(1,img_pad.shape[1]-1):
            #print(j)
            
            #parche 3x
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            #print(matriz1)
            for x in range(1,img_pad.shape[0]-1):
                for y in range(1,img_pad.shape[1]-1):
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                        
                        weights_ij = (np.exp(-distance/h_square))
                        
                        
                       # D_0 = 10
                        #alpha = 6
                        matriz_nu[x-1,y-1] = 1/(1+(np.abs(img_pad[i,j]-img_pad[x,y])/D_0)**(2*alpha))
                        #weights_ij_cpp = nu_ij*weights_ij
                
                        matriz_pesos[x-1,y-1] = weights_ij
                       
                    
            
            
            matriz_ponderada1 = matriz_pesos/np.sum(matriz_pesos)
            matriz_nu_pond = np.multiply(matriz_nu,matriz_ponderada1)
            matriz_ponderada_nu2 = matriz_nu_pond/np.sum(matriz_nu_pond)
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada_nu2))
       
    return matriz_imagen
 


def aniso_filter(img, iteraciones, grad):            

    #aplicamos sobel a una imagen con ruido
    img_sobel_g=filters.sobel(img)
    
    #realizamos el padding con img sobel
    img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')
    
    #matriz para almacenar 
    values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))
    #aplicamos algoritmo anisotropico
    cont=0
    row,col = img_sobel_g_pad.shape
    
    img_noisy_pad=np.pad(img, 1, mode='reflect') #padding de la original
    while cont<iteraciones:
        cont=cont+1
        
        for i in range(1,img_sobel_g_pad.shape[0]-1):
            for j in range(1,img_sobel_g_pad.shape[1]-1):
    
                parche_sobel = np.array([[img_sobel_g_pad[i-1,j-1],img_sobel_g_pad[i-1,j],img_sobel_g_pad[i-1,j+1]], 
                                       [img_sobel_g_pad[i,j-1],img_sobel_g_pad[i,j],img_sobel_g_pad[i,j+1]], 
                                        [img_sobel_g_pad[i+1,j-1],img_sobel_g_pad[i+1,j],img_sobel_g_pad[i+1,j+1]]])
                
                
                
                gradiente= np.sum(parche_sobel)
                    
                if gradiente<grad:
                    parche_noisy = np.array([[img_noisy_pad[i-1,j-1],img_noisy_pad[i-1,j],img_noisy_pad[i-1,j+1]], 
                                       [img_noisy_pad[i,j-1],img_noisy_pad[i,j],img_noisy_pad[i,j+1]], 
                                        [img_noisy_pad[i+1,j-1],img_noisy_pad[i+1,j],img_noisy_pad[i+1,j+1]]])
                    mean=np.mean(parche_noisy)
                    values[i-1, j-1]=mean
                else:
                    values[i-1, j-1]=img[i-1, j-1]

    return values















