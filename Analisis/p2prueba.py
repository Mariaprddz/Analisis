# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:35:21 2020

@author: nakag
"""

import scipy
import numpy as np
import nibabel as nib
from tkinter import *
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage import filters
from nilearn import datasets
from skimage.color import rgb2gray
from skimage.transform import resize 
#import modules

#%%

img = nib.load(r'\Users\Maria\Desktop\data\sub-01\anat\sub-01_T1w.nii.gz')


from nilearn import plotting

a = np.array(skimage.transform.resize(img.dataobj, (60,90)))


img_gray= a[:,:,128]  


#normalizo la imagen, el rango irá a partir de ahora [0,1], así que ojo a la hora de meter rangos en region growing
img_o=img_gray
img_o=img_o/np.max(img_o) 

#imagen con padding
img_pad = np.pad(img_o,1, mode='reflect')
#%%
#def RegionGrowingP2(img, rango_inferior, rango_superior):
    

plt.imshow(img_o, cmap='gray')
click_markers = plt.ginput(n=1,timeout=30)
def tuple_round(lista):
       try:
          return round(lista)
       except TypeError:
          return type(lista)(tuple_round(x) for x in lista)
     
markers = tuple_round(click_markers) 
seed = [(sub[1], sub[0]) for sub in markers] 

print(seed)


#if seed 

# [img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
#                                 [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
#                                 [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]
#%%
#Pasar nuestra a un array
pixels = np.array(seed)

#crear una matriz de ceros
region= np.zeros(shape=(img_o.shape[0],img_o.shape[1]))

#guardar nuestro pixel semilla en la ROI y lo llevamos a blanco
region[pixels[0,0],pixels[0,1]]=1
#definimos variables de umbral 
umbral_inf=0.2
umbral_sup=0.6

for i in range(1,img_pad.shape[0]-1):
        for j in range(1,img_pad.shape[1]-1):
            
            # Pixel names were chosen as shown:
            #
            #   -------------
            #   | a | b | c |
            #   -------------
            #   | d | e | f |
            #   -------------
            #   | g | h | k |
            #   -------------
            #
            # The current pixel is e
            # a, b, c, d,f,g,h,i are its neighbors of interest
            #
            # 255 is white, 0 is black
            # White pixels part of the background, so they are ignored
            # If a pixel lies outside the bounds of the image, it default to white
            #
            # If the current pixel is white, it's obviously not a component...
            
            if [i,j]==[pixels[0,0],pixels[0,1]]:
                pass
            
            elif img_pad[i-1, j+1] == img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i-1, j+1]=1
                
            elif img_pad[i-1, j]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i-1,j]=1
                
            elif img_pad[i-1,j-1]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i-1,j-1]=1
                
            elif img_pad[i,j-1]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i,j-1]=1
                
            elif img_pad[i,j+1]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i,j+1]=1
              
            elif img_pad[i+1,j-1]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i-1,j+1]=1
               
            elif img_pad[i+1,j]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i,j+1]=1
               
            elif img_pad[i+1,j+1]==img_pad[pixels[0,0]-umbral_inf,pixels[0,1]+umbral_sup]:
                region[i+1,j+1]=1
           
            
            
#%%%

coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
pixels = np.array(seed)
segment_size = 1
region = []
segmentation = np.zeros(img_o.shape)
rango_inferior=0.2
rango_superior = 0.6
segmentation[pixels[0,0],pixels[0,1]]=1
#%%
# for i in range(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 iterando sobre la imagen que vamos a ir comparando

#         for j in prange(1,img_pad.shape[1]-1):
            
#               #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
#               #Nótese que nuestro pixel central está en la posición [i,j]
#             matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
#                                 [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
#                                 [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
#%%
# pixels = np.array(seed)

# if pixel 
    
# def connected_adjacency(image, patch_size=(3, 3)):
#     r, c = image.shape[:2]
#     r = r / patch_size[0]
#     c = c / patch_size[1]
#     # constructed from 4 diagonals above the main diagonal
#     d1 = np.tile(np.append(np.ones(c-1), [0]), r)[:-1]
#     d2 = np.append([0], d1[:c*(r-1)])
#     d3 = np.ones(c*(r-1))
#     d4 = d2[1:-1]
#     upper_diags = s.diags([d1, d2, d3, d4], [1, c-1, c, c+1])
#     return upper_diags + upper_diags.T
#%%
# from skimage import measure
# import numpy as np

# # load array; arr = np.ndarray(...)
# # arr = np.zeros((10,10), dtype=bool)
# # arr[:2,:2] = True
# # arr[-4:,-4:] = True


# labeled = measure.label(segmentation, background=False, connectivity=2)
# print(labeled)
# label = labeled[8,8] # known pixel location
# print (label)

# rp = measure.regionprops(labeled)
# props = rp[label - 1] # background is labeled 0, not in rp

# props.bbox # (min_row, min_col, max_row, max_col)
# print(props.bbox)
# props.image # array matching the bbox sub-image
# print(props.image)
# #props.coordinates # list of (row,col) pixel indices

#%%
import sys
import math, random
from itertools import product

width=img_o.shape[0]
height=img_o.shape[1]


print(width,height)
regions=[]
for y, x in product(range(height), range(width)):
 
        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e | f |
        #   -------------
        #   | g | h | i |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, d,f,g,h,i are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #
        # If the current pixel is white, it's obviously not a component...
        if img_o[x, y] == 255:
            pass
 
        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and img_o[x, y-1] == 0:
            regions[x, y] = regions[(x, y-1)]
 
        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x+1 < width and y > 0 and img_o[x+1, y-1] == 0:
 
            c = regions[(x+1, y-1)]
            regions[x, y] = c
 
            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and img_o[x-1, y-1] == 0:
                a = regions[(x-1, y-1)]
                np.union(c, a)
 
            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and img_o[x-1, y] == 0:
                d = regions[(x-1, y)]
                np.union(c, d)
 
        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and img_o[x-1, y-1] == 0:
            regions[x, y] = regions[(x-1, y-1)]
 
        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and img_o[x-1, y] == 0:
            regions[x, y] = regions[(x-1, y)]
        
        
        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            pass
        
        
             