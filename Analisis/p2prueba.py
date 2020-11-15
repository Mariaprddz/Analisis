# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:35:21 2020

@author: nakag
"""

import scipy
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import skimage
from skimage import io
from skimage import filters
from nilearn import datasets
from skimage.color import rgb2gray
from skimage.transform import resize 
#import modules

#%%

img = nib.load(r'\Users\nakag\OneDrive\Escritorio\squizo\sub-02\anat\sub-02_T1w.nii.gz')


from nilearn import plotting

a = np.array(skimage.transform.resize(img.dataobj, (60,90)))


img_gray= a[:,:,128]  


#normalizo la imagen, el rango irá a partir de ahora [0,1], así que ojo a la hora de meter rangos en region growing
img_o=img_gray
img_o=img_o/np.max(img_o) 

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


# [img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
#                                 [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
#                                 [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]
#%%
coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
pixels = np.array(seed)
segment_size = 1
region = []
segmentation = np.zeros(img_o.shape)
rango_inferior=0.2
rango_superior = 0.6
segmentation[pixels[0,0],pixels[0,1]]=1

