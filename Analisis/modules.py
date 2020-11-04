# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 22:25:34 2020

@author: nakag
"""
#
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import scipy.ndimage.filters as filters
from numba import njit, prange

#Adición de ruido a las imágenes originales
'''----------------------Ruido Gaussiano-------------------'''
def add_gnoise(image,sigma):
    '''

    Parameters
    ----------
    image : Array of float32
        Imagen a la que queremos añadir ruido.
    sigma : float
        Parámetro que modula la cantidad de ruido gausiano que añadimos (desviación típica de la distribución normal).

    Returns
    -------
    noisy : Array of float64
        Imagen con ruido gaussiano.

    '''

    gaussian_noise=np.random.normal(loc=0.0, scale=sigma,size=np.shape(image))#Creación del ruido mediante una distribución normal a la que entra sigma como parámetro.
    noisy = image + gaussian_noise#Adición de ruido gaussiano a la imagen original
    return noisy

'''----------------------Ruido Impulsivo-------------------'''
def salpimienta(image,intensity):
    '''

    Parameters
    ----------
    image : Array of float32
        Imagen a la que queremos añadir ruido.
    intensity : float
        Parámetro que modula la cantidad de ruido impulsivo que añadimos.

    Returns
    -------
    ruido_output: Array of float64
        Imagen con ruido gaussiano.

    '''
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
    


def mean_filter(img,size_filter):

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


'''----------------------Filtro NLM-------------------'''    
#NLM: filtrar la imagen mediante el promedio ponderado de 
#los diferentes píxeles de la imagen en función de su similitud
@njit(parallel=True)
def nlm(img_ori,img_pad, h_square): 
    '''
    Parameters
    ----------
    img_ori : Array of float64
        Imagen ruidosa a filtrar.
    h_square : float
        Parámetro de similitud asociado al grado de filtrado.

    Returns
    -------
    matriz_imagen : Array of float64
        Imagen filtrada mediante NLM.

    '''

    #img_pad = np.pad(img_ori,1, mode='reflect') #Realizamos el padding de la imagen    
    
    #MAAAAAAL: matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #almacenamos los pesos en una matriz
    
    '''
    El problema principal por el que no funciona el código con numba tiene que
    ver con la matriz de pesos. Ésta hay que inicializarla justo antes de los 
    bucles x,y, no antes de i,j. Esto es debido a que para cada matriz1 se 
    debe crear una nueva matriz de pesos. 
    
    En la versión sin numba no pasa nada, ya que hasta que no se sobrescriben 
    todos los pesos en x,y no se pasa a filtrar la imagen. Sin embargo, con 
    numba no es así. Al paralelizar, la ejecución de los bucles x,y no se 
    produce de forma secuencial, sino que las iteraciones se van haciendo de 
    forma síncrona entre los hilos del procesador. ¿Qué ocurre? Que cuando un 
    hilo (núcleo) termina de hacer una comparación entre parches, pasa a hacer 
    otra comparación, pero... ¿qué matriz de pesos utiliza para guardar los 
    pesos nuevos? La última que se haya utilizado, y ésta puede ser una matriz 
    de pesos utilizada en el anterior filtrado (no es una matriz nueva 
    "limpia"). Por eso es importante crear la matriz de pesos antes de los 
    bucles x,y, para forzar que siempre que fijemos un parche nuevo, se utilice
    una matriz de pesos limpia.
    '''

    
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #Creamos otra matriz de ceros en la que indexaremos la imagen final

    for i in prange(1,img_pad.shape[0]-1):#Vamos a crear dos parches 3x3 que vamos a ir comparando 
        for j in prange(1,img_pad.shape[1]-1):
            
            #parche 3x3 de referencia (el que se quiere comparar con el resto de parches de la imagen, estará centrado en el pixel a filtrar)
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            
            # Aquí se inicializa la matriz de pesos!!
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #almacenamos los pesos en una matriz
            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                    #parche 3x3 que va recorriendo la imagen para compararse con el primero
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,j],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                    #Cálculo de la distancia euclídea
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                    #Ponderación de cada píxel en función de su similitud respecto al pixel a filtrar 
                        weights_ij = (np.exp(-distance/h_square)) #Nótese que el parámetro h_square va asociado al grado de filtrado
                    
                        matriz_pesos[x-1,y-1] = weights_ij #Introducimos cada uno de los peso a la matriz
                    
            
            #Ponderación de máscara obtenida (Aplicamos en este paso la cte de normalización Z(i))
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            #Finalmente se aplica la máscara a la imagen original
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen

'''----------------------Filtro NLM modificación 1-------------------'''   
#Función de NLM ponderando el parche original
@njit(parallel=True)
def nlm_samepatch(img_ori, img_pad, h_square):
    
    #padding 
   # img_pad = np.pad(img_ori,1, mode='reflect') #Realizamos el padding de la imagen original en el modo Reflect

    #matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
    
    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))

    for i in prange(1,img_pad.shape[0]-1):

        for j in prange(1,img_pad.shape[1]-1):
            
            #parche 3x
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                        
                        weights_ij = (np.exp(-distance/h_square))

                        matriz_pesos[x-1,y-1] = weights_ij 
                    
            matriz_pesos[i-1,j-1] = 0 #hago que nuestro pixel sea el de menor valor, para evitar que asuma que el máximo es la propia comparación consigo mismo
                        
            matriz_pesos[i-1,j-1] = np.max(matriz_pesos) #obtenemos el valor máximo de similitud que se haya encontrado en el resto de la imagen
            
            
            matriz_ponderada = matriz_pesos/np.sum(matriz_pesos)
            
            
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada))
       
    return matriz_imagen


'''----------------------Filtro NLM-cpp modificación 2-------------------'''  
@njit(parallel=True)
def nlm_cpp(img_ori, img_pad, h_square, D_0, alpha):

    #img_pad = np.pad(img_ori,1, mode='reflect') 
    
    #matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
    #matriz_nu = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #esta será la matriz que multiplicaremos por los pesos normalizados

    matriz_imagen = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1]))
    
    for i in prange(1,img_pad.shape[0]-1):
        for j in prange(1,img_pad.shape[1]-1):
            
            #parche 3x3
            
            matriz1 = np.array([[img_pad[i-1,j-1],img_pad[i-1,j],img_pad[i-1,j+1]], 
                                [img_pad[i,j-1],img_pad[i,j],img_pad[i,j+1]], 
                                [img_pad[i+1,j-1],img_pad[i+1,j],img_pad[i+1,j+1]]])
            
            '''
            Al igual que pasa con matriz_pesos, matriz_nu se debe inicializar
            antes de los bucles x,y, no antes de i,j.
            '''
            
            matriz_pesos = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #aqui almacenamos los pesos
            matriz_nu = np.zeros(shape=(img_ori.shape[0],img_ori.shape[1])) #esta será la matriz que multiplicaremos por los pesos normalizados

            for x in prange(1,img_pad.shape[0]-1):
                for y in prange(1,img_pad.shape[1]-1):
                    
                        matriz2 = np.array([[img_pad[x-1,y-1],img_pad[x-1,y],img_pad[x-1,y+1]], 
                                [img_pad[x,y-1],img_pad[x,y],img_pad[x,y+1]], 
                                [img_pad[x+1,y-1],img_pad[x+1,y],img_pad[x+1,y+1]]])
                    
                        distance = np.sqrt((matriz1-matriz2)**2)
                        distance = np.sum(distance)
                        
                        weights_ij = (np.exp(-distance/h_square))
                        
                        matriz_nu[x-1,y-1] = 1/(1+(np.abs(img_pad[i,j]-img_pad[x,y])/D_0)**(2*alpha))
                
                        matriz_pesos[x-1,y-1] = weights_ij
                           
            
            matriz_ponderada1 = matriz_pesos/np.sum(matriz_pesos)#normalización de los pesos
            
            matriz_nu_pond = np.multiply(matriz_nu,matriz_ponderada1)#ponderamos los pesos por nu, para que dependan de la similitud entre píxeles centrales
            
            matriz_ponderada_nu2 = matriz_nu_pond/np.sum(matriz_nu_pond)#normalización de los pesos tras ponderar por nu
           
            matriz_imagen[i-1,j-1] = np.sum(np.multiply(img_ori,matriz_ponderada_nu2))
       
    return matriz_imagen
 

'''----------------------Filtro Anisotrópico-------------------'''  

def aniso_filter(img, iteraciones, threshold):            
    cont=0
    if cont==0:
        #aplicamos sobel a una imagen con ruido
        img_sobel_g=filters.sobel(img)
        
        #realizamos el padding con img sobel
        img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')
        
        #matriz para almacenar 
        values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))
        #aplicamos algoritmo anisotropico
        cont=0
        
        img_noisy_pad=np.pad(img, 1, mode='reflect') #padding de la original
    else:
                #aplicamos sobel a una imagen con ruido
        img_sobel_g=filters.sobel(values)
        
        #realizamos el padding con img sobel
        img_sobel_g_pad=np.pad(img_sobel_g, 1, mode='reflect')
                
        img_noisy_pad=np.pad(values, 1, mode='reflect') #padding de la original
        #matriz para almacenar 
        values=np.zeros(shape=(img_sobel_g.shape[0],img_sobel_g.shape[1]))

        
        img_noisy_pad=np.pad(img, 1, mode='reflect') #padding de la original
    while cont<iteraciones:
        
        '''
        Fallo importante. El número de iteraciones indica el número de veces
        que queréis que se repita este proceso. Intuitivamente, lo que podríais
        pensar es que, a más iteraciones, más suavizado. Sin embargo, tal y como
        tenéis el código, da igual el número de iteraciones que pongáis, porque 
        siempre va a salir lo mismo.  ¿Dónde está el fallo? En el hecho de que
        en los bucles debéis introducir a la entrada la imagen de salida que 
        habéis obtenido en la iteración inmediatamente anterior. Por tanto: lo 
		que hay que hacer es suavizar la imagen que se ha suavizado en la 
		iteración anterior.
        
        Dicho de otra forma: las variables img_sobel_g_pad e img_noisy_pad las
        debéis volver a calcular con la matriz values que sacáis al final.
        '''

        for i in range(1,img_sobel_g_pad.shape[0]-1):
            for j in range(1,img_sobel_g_pad.shape[1]-1):
    
                parche_sobel = np.array([[img_sobel_g_pad[i-1,j-1],img_sobel_g_pad[i-1,j],img_sobel_g_pad[i-1,j+1]], 
                                       [img_sobel_g_pad[i,j-1],img_sobel_g_pad[i,j],img_sobel_g_pad[i,j+1]], 
                                        [img_sobel_g_pad[i+1,j-1],img_sobel_g_pad[i+1,j],img_sobel_g_pad[i+1,j+1]]])
                
                
                
                gradiente= np.sum(parche_sobel)
                    
                if gradiente<threshold:
                    parche_noisy = np.array([[img_noisy_pad[i-1,j-1],img_noisy_pad[i-1,j],img_noisy_pad[i-1,j+1]], 
                                       [img_noisy_pad[i,j-1],img_noisy_pad[i,j],img_noisy_pad[i,j+1]], 
                                        [img_noisy_pad[i+1,j-1],img_noisy_pad[i+1,j],img_noisy_pad[i+1,j+1]]])
                    mean=np.mean(parche_noisy)
                    values[i-1, j-1]=mean
                else:
                    values[i-1, j-1]=img[i-1, j-1]
        
        cont=cont+1


    return values