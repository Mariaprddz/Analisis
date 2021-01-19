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
 