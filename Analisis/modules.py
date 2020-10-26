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
    size_filter = 5
    # the filter is divided by size_filter^2 for normalization
    mean_filter = np.ones((size_filter,size_filter))/np.power(size_filter,2)
    # performing convolution
    img_meanfiltered = filters.convolve(img, mean_filter,mode='reflect')
    return img_meanfiltered


def median_filter(img):
    img_filtered = filters.median_filter(img,size=5,mode='reflect')
    return img_filtered

def gaussian_filter(img):

    # performing convolution
    img_gaussianfiltered = filters.gaussian_filter(img, sigma = 3,mode='reflect')
    return img_gaussianfiltered

def submatrix(img, row, col):

    for i in range(0,row-2):
        #print(i)
        for j in range(0,col-2):
            #print(j)
            
            #parche 3x
            
            matrix1 = np.array([[img[i-1,j-1],img[i-1,j],img[i-1,j+1]], 
                                [img[i,j-1],img[i,j],img[i,j+1]], 
                                [img[i+1,j-1],img[i+1,j],img[i+1,j+1]]])
            
            #print(matrix1)
    
            
            for x in range(0,row-2):
                for y in range(0,col-2):
                    
                    matrix2 = np.array([[img[x-1,y-1],img[x-1,y],img[x-1,y+1]], 
                                [img[x,y-1],img[x,j],img[x,y+1]], 
                                [img[x+1,y-1],img[x+1,y],img[x+1,y+1]]])
                    
                    distance = np.sqrt((matrix1-matrix2)**2)
                    distance = np.sum(distance)
                    
                   # print(matrix2)
                    
                    #print(distance)
                    
        return distance

    
    




















