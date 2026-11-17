# dépendances.
# pip install opencv-python numpy matplotlib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("image.jpg")

# conversion pour afficher en matplotlib
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# et gris
image_gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# TODO Créer un noyau de convolution 3x3 pour un filtre moyenneur
noyau_moyenneur = np.ones((3, 3), np.float32) / 9

# Appliquer le filtre moyenneur.
'''
image_filtrée = cv.filter2D(image_gris, -1, noyau_moyenneur)
cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', image_filtrée)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# Recommencer avec des noyaux 5x5 et 9x9.
'''
noyau_5x5 = np.ones((5, 5), np.float32) / 25
noyau_9x9 = np.ones((9, 9), np.float32) / 81

image_filtrée_5x5 = cv.filter2D(image_gris, -1, noyau_5x5)
cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', image_filtrée_5x5)
cv.waitKey(0)
cv.destroyAllWindows()
'''

'''
image_filtrée_9x9 = cv.filter2D(image_gris, -1, noyau_9x9)
cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', image_filtrée_9x9)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# Comparer les résultats.
# les images filtrées sont plus lisses avec des noyaux plus grands.


# bruitage d'une image
bruit = np.random.normal(0, 25, image_gris.shape)

image_bruitee = (image_gris.astype(np.float32) + bruit)

# TODO Afficher l'image bruitée.
'''
cv.namedWindow('image bruitée', cv.WINDOW_NORMAL)
cv.imshow('image bruitée', image_bruitee)
cv.waitKey(0)
cv.destroyAllWindows()
'''
# Appliquer un filtre gaussien.
'''
image_gaussienne = cv.GaussianBlur(image_bruitee, (5, 5), 0)
cv.namedWindow('image gaussienne', cv.WINDOW_NORMAL)
cv.imshow('image gaussienne', image_gaussienne)
cv.waitKey(0)
cv.destroyAllWindows()
'''
# Appliquer un filtre médian.
'''
image_mediane = cv.medianBlur(image_bruitee, 5)
cv.namedWindow('image mediane', cv.WINDOW_NORMAL)
cv.imshow('image mediane', image_mediane)
cv.waitKey(0)
cv.destroyAllWindows() 
'''
# Comparer les résultats (flou, contours, etc).
# le filtre gaussien est plus efficace pour réduire le bruit, tandis que le filtre médian est plus efficace pour préserver les contours.

 


# TODO Créer le noyau Laplacien.
#       |  0  -1   0 |
#       | -1   4  -1 |
#       |  0  -1   0 |
'''
noyau_laplacien = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
# Appliquer le filtre.
image_laplacien = cv.filter2D(image_gris, -1, noyau_laplacien)
# Afficher le résultat.
cv.namedWindow('image laplacien', cv.WINDOW_NORMAL)
cv.imshow('image laplacien', image_laplacien)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# TODO : donner la formule de cv2.Sobel (en cherchant sur internet) et expliquer en quoi cela donne les contours de l'image
# la fonction cv2.Sobel calcule la dérivée première de l'image dans une direction donnée (x ou y) en utilisant un noyau de convolution. Cela permet de détecter les variations d'intensité dans l'image, ce qui correspond aux contours.


# Le gradient total est donné par :
# G = sqrt(Gx^2 + Gy^2) avec Gx et Gy les variations verticales et horizontales. 
# calculer et afficher le gradient total. 
'''
Gx = cv.Sobel(image_gris, cv.CV_64F, 1, 0, ksize=5)
Gy = cv.Sobel(image_gris, cv.CV_64F, 0, 1, ksize=5)
G = np.sqrt(Gx**2 + Gy**2)
G = np.uint8(G) 
cv.namedWindow('gradient total', cv.WINDOW_NORMAL)
cv.imshow('gradient total', G)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# Quelle est la différence entre un filtre passe-bas et un filtre passe-haut ?
# le filtre passe-bas laisse passer les basses fréquences (les variations lentes de l'image) et atténue les hautes fréquences (les variations rapides de l'image), tandis que le filtre passe-haut fait l'inverse, laissant passer les hautes fréquences et atténuant les basses fréquences.

# Étudier la documentation du filtre de Canny pour comprendre ses paramètres.
image_canny = cv.Canny(image_gris, 100,200)
cv.namedWindow('image canny', cv.WINDOW_NORMAL)
cv.imshow('image canny', image_canny)
cv.waitKey(0)       
cv.destroyAllWindows()

#les paramètres du filtre de Canny sont les seuils inférieur et supérieur pour la détection des contours. Le seuil inférieur est utilisé pour détecter les contours faibles, tandis que le seuil supérieur est utilisé pour détecter les contours forts. Les contours faibles sont connectés aux contours forts pour former des contours complets.


''' Pour ceux qui ont fini trop tôt :

Générer un bruit poivre et sel en perturbant aléatoirement 10% des pixels en leur mettant pour valeur 0 our 255. 

Étudier les performances d'uen filtre Gaussien et d'un filtre médian. 

Lire la documentation de la fonction PSNR et tester les performances de différents filtre. 

Implémenter manuellement (avec des boucles for) un filtre moyenneur sans utiliser cv2.filter2D() et comparer le temps d'exécution entre une implémentation manuelle et OpenCV.

Étudier l'influence de la taille du noyau sur le temps d'exécution.
'''