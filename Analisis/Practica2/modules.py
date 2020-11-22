# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:35:21 2020

@author: nakag
"""

import scipy
import numpy as np
import nibabel as nib

import matplotlib

import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage import measure
from skimage import filters
from nilearn import datasets
from skimage.color import rgb2gray
from skimage.transform import resize 
from imimposemin import imimposemin
from skimage.segmentation import watershed

    

def RegionGrowingP2(img, umbral_inf, umbral_sup):

    plt.imshow(img, cmap='gray')
    click_markers = plt.ginput(n=1,timeout=30)
    click_markers = list(click_markers[0])
    print(click_markers)
         
    markers = [round(num) for num in click_markers ]
    seed = [markers[1],markers[0]] 
     
    
    print(seed)
    
    coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
    
    #Pasar nuestra a un array
    pixels = np.array(seed)
    
    #crear una matriz de ceros
    region= np.zeros(shape=(img.shape[0],img.shape[1]))
    
    #guardar nuestro pixel semilla en la ROI y lo llevamos a blanco
    region[pixels[0],pixels[1]]=1
    #definimos variables de umbral 
    #umbral_inf=0.1
    #umbral_sup=0.1
    intervalo_inf = img[pixels[0],pixels[1]]-umbral_inf
    intervalo_sup = img[pixels[0],pixels[1]]+umbral_sup
    
    
    for x in range (0, coords.shape[0]):
    
        if intervalo_inf<=img[pixels[0]+coords[x,0], pixels[1]+coords[x,1]]<=intervalo_sup :
                                
            region[pixels[0]+coords[x,0], pixels[1]+coords[x,1]] = 1      
            
        else:
            pass
                    
    new_pix = np.where(region == 1)
    Coordinates = np.array(list(zip(new_pix[0], new_pix[1])))
    listOfCoordinates = list(zip(new_pix[0], new_pix[1]))
    regionCoords =[]
    
                
    while len (listOfCoordinates)!=len(regionCoords):
        new_pix = np.where(region == 1)
        Coordinates = np.array(list(zip(new_pix[0], new_pix[1]))) 
        listOfCoordinates = list(zip(new_pix[0], new_pix[1]))
        
        for i in range (0, Coordinates.shape[0]):
        
                for x in range (0, 8):
                    if Coordinates[i,0]+coords[x,0] >= 0 and Coordinates[i,1]+coords[x,1]>= 0 and Coordinates[i,0]+coords[x,0]<img.shape[0] and Coordinates[i,1]+coords[x,1]<img.shape[1]:
                        if intervalo_inf<=img[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]]<=intervalo_sup :
                                        
                            region[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]] = 1      
                            
                        else:
                            pass
                    else:
                        pass
        regionCoords = np.where(region == 1)
        regionCoords = list(zip(regionCoords[0], regionCoords[1]))

    
    return region            

            

def WatershedExerciseP2(img, numberofseeds):
    
    white_dots= np.zeros(shape=(img.shape[0],img.shape[1]))
    img_sobel=filters.sobel(img)  
              
    plt.figure()
    plt.title('sobelsito')
    plt.imshow(img_sobel, cmap=plt.cm.gray)
    
    plt.title('semillitas')
    plt.imshow(img, cmap='gray')
    click_markers = plt.ginput(n=numberofseeds)
    clicks = [(sub[1], sub[0]) for sub in click_markers]
    markers = np.array(clicks,dtype = int)
    
    print(markers)

    
    white_dots[markers[:,0], markers[:,1]] = 1
    plt.title('mascarita binaria')
    plt.imshow(white_dots, cmap=plt.cm.gray)    
    
    minimos = imimposemin(img_sobel, white_dots)     #modifica la imagen de la máscara en escala de grises utilizando la reconstrucción morfológica por lo que sólo tiene mínimo regional donde la imagen de marcador binario es distinto de cero.IBW
    
    watershed1= watershed(img_sobel)
    watershed2 = watershed(minimos)
    
    return watershed1, watershed2







