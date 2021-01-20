import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import scipy.ndimage.filters as filters
from numba import njit, prange


'''----------------------Filtro NLM-cpp modificación 2-------------------'''  
@njit(parallel=True)
def nlm_cpp(img_ori, img_pad, h_square, D_0, alpha):
    '''
    Parameters
    ----------
    img_ori : Array of float64
        Imagen ruidosa a filtrar.
    img_pad: Array of float64
        Imagen ruidosa con padding.
    h_square : float
        Parámetro de similitud asociado al grado de filtrado.
    D_0 : float
        Parámetro de la función eta que calcula la similitud entre píxeles
    alpha : float
        Parámetro de la función eta que calcula la similitud entre píxeles
    Returns
    -------
    matriz_imagen : Array of float64
        Imagen filtrada mediante NLM (comparación de píxeles centrales).
    '''

    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))#Creamos otra matriz de ceros en la que indexaremos la imagen final
    
    for i in prange(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 iterando sobre la imagen que vamos a ir comparando
        for j in prange(1,img_pad.shape[1]-1):
            
             #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
              #Nótese que nuestro pixel central está en la posición [i,j]            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            '''
            Al igual que pasa con matriz_pesos, matriz_nu se debe inicializar
            antes de los bucles x,y, no antes de i,j.
            '''
            # Aquí se inicializa la matriz de pesos y la matriz nu para que no surjan errores al implementar numba
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
            matriz_nu = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #esta será la matriz que multiplicaremos por los pesos normalizados

            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                    #parche 3x3 que va recorriendo la imagen para compararse con el primero
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                    #Cálculo de la distancia euclídea
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    #Ponderación de cada píxel en función de su similitud respecto al pixel a filtrar                    
                        weights_ij = (np.exp(-distance/h_square))#Nótese que el parámetro h_square va asociado al grado de filtrado
                    #Ponderación de similitud entre píxeles centrales   
                        matriz_nu[x-1,y-1] = 1/(1+(np.abs(img_pad[i,j]-img_pad[x,y])/D_0)**(2*alpha))
                
                        matriz_pesos[x-1,y-1] = weights_ij#Introducimos cada uno de los peso a la matriz
                           
            #Ponderación de máscara obtenida (Aplicamos en este paso la cte de normalización Z(i))
            matriz_ponderada1 = matriz_pesos/np.sum(matriz_pesos)#normalización de los pesos
            
            matriz_nu_pond = np.multiply(matriz_nu,matriz_ponderada1)#ponderamos los pesos por nu, para que dependan de la similitud entre píxeles centrales
            
            matriz_ponderada_nu2 = matriz_nu_pond/np.sum(matriz_nu_pond)#normalización de los pesos tras ponderar por nu
            #Finalmente se aplica la máscara a la imagen original
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada_nu2))
       
    return matriz_imagen



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



    
#crecimiento de regiones
def RegionGrowingP2(img, umbral_inf, umbral_sup):
    '''
    Parameters
    ----------
    img : Array of float32
        Imagen original
    umbral_inf : float
        Umbral por debajo del nivel de gris del punto seleccionado en seed.
    umbral_sup : float
        Umbral por encima del nivel de gris del punto seleccionado en seed.
    Returns
    -------
    region : Array of float64
        ROI de la imagen deseada.
    '''
    plt.figure()
    plt.imshow(img, cmap='gray') #Hacemos la representación la imagen para poder decidir donde posicionar la semilla
    click_markers = plt.ginput(n=1)  #Utilizamos la funcion .ginput() para posicionamiento de semilla
    plt.close()
    click_markers = list(click_markers[0]) #Transformamos a una lista

         
    markers = [round(num) for num in click_markers ] #Redondeamos a números enteros
    seed = [markers[1],markers[0]] #Cambiamos el orden de los elementos para poder utilizarlo como coordenadas
     
    
    print('Las coordenadas de las semillas son: ', seed)
    
    coords = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]) #Lista de coordenadas para hacer las comparaciones con los píxeles adyacentes
    
    #Pasar nuestra semilla a un array
    pixels = np.array(seed)
    
    #crear una matriz de ceros
    region= np.zeros(shape=(img.shape[0],img.shape[1]))
    
    #guardar nuestro pixel semilla en la ROI y lo llevamos a blanco
    region[pixels[0],pixels[1]]=1
    #definimos nuestro intervalo para decidir si incluirlo en la ROI o no

    intervalo_inf = img[pixels[0],pixels[1]]-umbral_inf
    intervalo_sup = img[pixels[0],pixels[1]]+umbral_sup
    
    #Parseamos por la lista coords para ir haciendo la conectividad a 8.
    for x in range (0, coords.shape[0]):
        #Dfinimos una condicion en la que si la intensidad del pixel comparado se encuentra entre el intervalo de similitud
        if intervalo_inf<=img[pixels[0]+coords[x,0], pixels[1]+coords[x,1]]<=intervalo_sup :
                                
            region[pixels[0]+coords[x,0], pixels[1]+coords[x,1]] = 1     #Se añade a la ROI sustituyendo en dichas coordenadas por un 1  (se lleva a blanco)
            
        else:
            pass #Si no cumple la condición se prosigue con la comparación
                    
    new_pix = np.where(region == 1) #Buscamos aquellas coordenadas en las que la matriz region es igual a 1
    Coordinates = np.array(list(zip(new_pix[0], new_pix[1]))) #Creamos un array con dichas coordenadas
    listOfCoordinates = list(zip(new_pix[0], new_pix[1])) #lista de las coordenadas
    regionCoords =[] #creamos una lista vacía para realizar la comparación
    
                
    while len (listOfCoordinates)!=len(regionCoords): #COmparamos las listas de coordenadas de los puntos en los que la región es 1, antes y después de realizar cada iteración de conectividad a 8
        new_pix = np.where(region == 1)#Buscamos aquellas coordenadas en las que la matriz region es igual a 1
        Coordinates = np.array(list(zip(new_pix[0], new_pix[1])))  #Creamos un array con dichas coordenadas
        listOfCoordinates = list(zip(new_pix[0], new_pix[1])) #lista de las coordenadas
        
        for i in range (0, Coordinates.shape[0]): #Para todas las coordenadas de los píxeles de la región creamos un bucle for
        
                for x in range (0, 8):
                    if Coordinates[i,0]+coords[x,0] >= 0 and Coordinates[i,1]+coords[x,1]>= 0 and Coordinates[i,0]+coords[x,0]<img.shape[0] and Coordinates[i,1]+coords[x,1]<img.shape[1]: #Creamos una condición para evitar que el algoritmo de error al comparar píxeles de los bordes de la imagen. 
                    #Esto lo conseguimos imponiendo que las comparaciones no se hagan sobre coordenadas negativas ni fuera de rango
                        if intervalo_inf<=img[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]]<=intervalo_sup :
                                        
                            region[Coordinates[i,0]+coords[x,0], Coordinates[i,1]+coords[x,1]] = 1      
                            #Volvemos a hacer la misma comparación que en la primera iteración explicada fuera del bucle while
                        else:
                            pass
                    else:
                        pass
        regionCoords = np.where(region == 1)#Volvemos a evaluar la nueva roi con cada iteración.
        regionCoords = list(zip(regionCoords[0], regionCoords[1]))#Convertimos a lista

    #Devolvemos la ROI final como resultado de la función
    return region            

         

