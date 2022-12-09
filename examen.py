import numpy as np
import cv2 
import math
import matplotlib.pyplot as plt
import random
from math import dist

#########################Sacaremos los clusters de los jitomates###########################
# def euclidean_distance(x1, x2): # Calcula la distincia euclidiana 
#     # return np.sqrt(np.sum((x1 - x2)**2))
#     return distance.euclidean(x1, x2)

# def kmeans(k, max_iterations, image, pixels):
#     pixel_values = image.reshape((-1, 3)).copy()
#     centroids = []

#     random.seed(50)
#     for _ in range(k):
#         pixel_random = random.randint(0, pixels)
#         print("pixeles random: ", pixel_random)
#         centroids.append(pixel_values[pixel_random])
#     print("Centroides: ", centroids)

#     for adj in range(max_iterations):
#         print("No.iteraciones: ", adj+1)
#         clusters = [[] for _ in range(k)]
#         for idx, valuePixel in enumerate(pixel_values):
#             distances = [euclidean_distance(valuePixel, point) for point in centroids]
#             closest_index = np.argmin(distances)
#             clusters[closest_index].append(idx)

#         centroidsOld = centroids 
#         centroids = np.zeros((k, 3)) 
#         for cluster_idx, cluster in enumerate(clusters):
#             cluster_mean = np.mean(pixel_values[cluster], axis=0)
#             centroids[cluster_idx] = cluster_mean 

#         distances = [euclidean_distance(centroidsOld[i], centroids[i]) for i in range(k)]
#         distances = sum(distances)
#         if distances == 0:
#             break

#     for clusterIndex, cluster in enumerate(clusters): # Ultimo cluster
#         pixel_values[cluster] = np.uint8(centroids[clusterIndex])
#     imageClustering = pixel_values.reshape(image.shape)

#     #Quitamos cluster que no necesitamos
#     clusterquitar = [0, 2, 3]
#     imageSeg = image.reshape((-1, 3)).copy() 
#     for i in clusterquitar:
#         imageSeg[clusters[i]] = np.uint8(np.array([0, 0, 0]))
#     imageSeg = imageSeg.reshape(image.shape)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     imageClustering = cv2.cvtColor(imageClustering, cv2.COLOR_BGR2RGB)
#     imageSeg = cv2.cvtColor(imageSeg, cv2.COLOR_BGR2RGB)

#     fig, axs = plt.subplots(1, 3, figsize=(20, 10))
#     axs[0].imshow(image)
#     axs[1].imshow(imageClustering)
#     axs[2].imshow(imageSeg)

#     for a in axs:
#         a.set_axis_off()

#     plt.show()

# image = cv2.imread("Jit2.png") # Lee imagen
# rows = image.shape[0]
# cols = image.shape[1]
# pixels = rows * cols # Pixeles totales
# print("Tamaño imagen: " + str(rows) + " x " + str(cols))
# print("Total pixeles:", pixels, "\n")

# k = 4 
# maxIterations = 220
# kmeans(k, maxIterations, image, pixels)




############Kmeans con funcion de python###############
image = cv2.imread("Jit1.jpg") # Lee imagen

valores_pixeles = np.float32(image.reshape((-1, 3)).copy())
criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, etiquetas, centros = cv2.kmeans(valores_pixeles, 4, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)

centro = np.uint8(centros)
imageClustering = centro[etiquetas.flatten()]
imageClustering = imageClustering.reshape((image.shape))

imageSeg = image.reshape((-1, 3)).copy()
clusterquitar = [0, 1, 2]
for i in clusterquitar:
    imageSeg[etiquetas.flatten() == i] = np.uint8(np.array([0, 0, 0]))
imageSeg = imageSeg.reshape(image.shape)

#Binariza la imagen obtenida por kmeans
def binarizar(image):
    imageBin = np.zeros((image.shape[0], image.shape[1]))
    rows, cols = image.shape[0], image.shape[1]
    #rojos = np.array([142, 36, 30])
    #rojos = np.array([141, 36, 29])
    rojos = np.array([142, 36, 30])
    for i in range(rows):
        for j in range(cols):
            if np.array_equal(image[i, j], rojos):
                imageBin[i,j] = 255
            else:
                imageBin[i,j] = 0
    return imageBin


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageClustering = cv2.cvtColor(imageClustering, cv2.COLOR_BGR2RGB)
imageSeg = cv2.cvtColor(imageSeg, cv2.COLOR_BGR2RGB)
plt.title("ImgClustering")
plt.imshow(imageClustering)
plt.show()


binarizada = binarizar(imageClustering)
binarizada=np.uint8(binarizada)

#Encuentra los contornos utilizando la binarizada
contornos , hierarchy = cv2.findContours(binarizada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Quita contornos que tengan longitud menor a 100 (ruidos)
index = []
for i in range(len(contornos)):
    if len(contornos[i]) < 700:
        index.append(i)
        
contornosNuevos = np.delete(contornos, index)
#print("coordenadas Jit1: ", contornosNuevos[2])

# print("contornos: ", contornos[0])
binarizada = cv2.cvtColor(binarizada, cv2.COLOR_BGR2RGB)
plt.title("Binarizada")
plt.imshow(binarizada)
plt.show()

#Copias para dibujar los bordes
contornoJ1 = np.copy(image)  #se dibujan todos los bordes
contornoJ2 = np.copy(image) #se dibuja linea para el jitomate1
imageCpy = np.copy(image) #se dibuja linea para el jitomate2
imageFinal = np.copy(image)

#Dibuja todos los contornos
cv2.drawContours(imageCpy, contornosNuevos, -1, (0,0,255), 3)
plt.title("Img con todos los contornos")
plt.imshow(imageCpy)
plt.show()

#Dibujar los contornos de un jitomate
cv2.drawContours(contornoJ1, contornosNuevos, 2, (0,0,255), 3)
plt.title("Contorno Jitomate2")
plt.imshow(contornoJ1)
plt.show()

#Dibujar los contornos de otro jitomate
cv2.drawContours(contornoJ2, contornosNuevos, 0, (0,0,255), 3)
plt.title("Contorno Jitomate4")
plt.imshow(contornoJ2)
plt.show()
#Distancias 
# def distance(x ,y):

#Encuentra 4 puntos que nos ayuden a identificar los extremos de los jitomates
def puntosRecta1(coordenadas, image):
    TotalCoordenadas = len(coordenadas)

    mit = TotalCoordenadas // 2
    oct = TotalCoordenadas // 8

    #Obtengo 4 puntos que son contrarios
    x1, y1 = coordenadas[TotalCoordenadas - oct][0]
    x2, y2 = coordenadas[mit - oct][0]

    img = cv2.line(image, (x1,y1), (x2,y2), (225,225,0), 4)

    #Distancia linea1
    print("Datos Jitomate 4")
    print("Puntos de medicion:")
    print("Punto 1: ", str(x1) + ", " + str(y1))
    print("Punto 2: ", str(x2) + ", " + str(y2))
    distance1 = round(dist((x1,y1), (x2,y2)))
    print("distanciaJitomate4: ", str(distance1) + "pixles")

    return img 

def puntosRecta2(coordenadas, image):
    TotalCoordenadas = len(coordenadas)

    mit = TotalCoordenadas // 2
    oct = TotalCoordenadas // 8
    doce = TotalCoordenadas // 9 #8 #10 #11

    #Obtengo 4 puntos que son contrario
    x1, y1 = coordenadas[TotalCoordenadas - oct - doce][0]
    x2, y2 = coordenadas[mit - oct - doce][0]

    img = cv2.line(image, (x1,y1), (x2,y2), (225,225,0), 4)

    print("\nDatos Jitomate 2")
    print("Puntos de medicion:")
    print("Punto 1: ", str(x1) + ", " + str(y1))
    print("Punto 2: ", str(x2) + ", " + str(y2))
    distance2 = round(dist((x1,y1), (x2,y2)))
    print("distanciaJitomate2: ", str(distance2) + "pixels")

    return img

imgRecta1= puntosRecta1(contornosNuevos[0], imageFinal)
# plt.imshow(imgRecta1)
# plt.show()

imgRecta2= puntosRecta2(contornosNuevos[2], imageFinal)
rows, cols = imgRecta2.shape[0] , imgRecta2.shape[1]
# print("Tamaño: ", rows, cols)
# plt.title("Img Final")
plt.imshow(imgRecta2)
plt.show()







