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
print(click_markers[0])
click_markers = list(click_markers[0])
print(click_markers)
     
markers = [round(num) for num in click_markers ]
seed = [markers[1],markers[0]] 
 

print(seed)


# [img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
#                                 [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
#                                 [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]
#%%
coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])

    
#%%
#Pasar nuestra a un array
pixels = np.array(seed)

#crear una matriz de ceros
region= np.zeros(shape=(img_o.shape[0],img_o.shape[1]))

#guardar nuestro pixel semilla en la ROI y lo llevamos a blanco
region[pixels[0],pixels[1]]=1
#definimos variables de umbral 
umbral_inf=0.2
umbral_sup=0.1
intervalo_inf = img_o[pixels[0],pixels[1]]-umbral_inf
intervalo_sup = img_o[pixels[0],pixels[1]]+umbral_sup



    

print (img_o.shape)                

for x in range (0, coords.shape[0]):

    if intervalo_inf<=img_o[pixels[0]+coords[x,0], pixels[1]+coords[x,1]]<=intervalo_sup :
                            
        region[pixels[0]+coords[x,0], pixels[1]+coords[x,1]] = 1      
        
    else:
        pass
                
new_pix = np.where(region == 1)
listOfCoordinates = np.array(list(zip(new_pix[0], new_pix[1])))
print(listOfCoordinates[0,1])
    
for i in range (0, listOfCoordinates.shape[0]):

        for x in range (0, 8):

            if intervalo_inf<=img_o[listOfCoordinates[i,0]+coords[x,0], listOfCoordinates[i,1]+coords[x,1]]<=intervalo_sup :
                            
                region[listOfCoordinates[i,0]+coords[x,0], listOfCoordinates[i,1]+coords[x,1]] = 1      
        
            else:
                pass
    
 #%%
  

