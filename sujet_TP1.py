import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')  # Évite les erreurs d'affichage graphique


# === Étape 1 : Lecture et affichage de l'image ===
# -------------------------------------------------
# Lecture de l'image
img = cv.imread("tulipes.jpg")

# Vérification que l'image est bien chargée
if img is None:
    print("⚠️ Erreur : image introuvable ! Vérifiez le chemin du fichier.")
else:
    cv.namedWindow('image originale', cv.WINDOW_NORMAL)
    cv.imshow('image originale', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# === Étape 2 : Filtrage bilatéral ===
# ------------------------------------
# TODO : Appliquez un filtrage bilatéral à l’image.

# Quelle est la différence entre un flou bilatéral et un flou gaussien ?
# flou bilatéral préserve les contours de l'image tout en réduisant le bruit, tandis que le flou gaussien lisse l'image sans tenir compte des contours. 

# Quels sont les trois paramètres principaux du filtre ?
# Les trois paramètres principaux du filtre bilatéral sont :
# - d : diamètre du noyau de filtrage
# - sigmaColor : écart-type de la couleur
# - sigmaSpace : écart-type de l'espace

filtered = cv.bilateralFilter(img, d=20, sigmaColor=100, sigmaSpace=100)
cv.namedWindow('image filtrée', cv.WINDOW_NORMAL)
cv.imshow('image filtrée', filtered)
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 3 : Conversion en niveaux de gris ===
# -----------------------------------------------
# Conversion de l’image couleur en niveaux de gris
# TODO : utilisez cv.cvtColor avec le bon code de conversion.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.namedWindow('image grise', cv.WINDOW_NORMAL)
cv.imshow('image grise', gray)
cv.waitKey(0)
cv.destroyAllWindows()

# QUESTION : Comment est effectuée la conversion en niveaux de gris concrètement ?
# la conversion en niveaux de gris est effectuée en combinant les canaux R, G et B

# Donnez la formule mathématique qui combine les canaux R, G et B :
# Y = 0.299 * R + 0.587 * G + 0.114 * B


# === Étape 4 : Seuillage binaire simple ===
# ------------------------------------------
# TODO : Appliquez un seuillage simple avec une valeur de 150.

# Que signifie cette opération sur les pixels de l’image ?
# cette opération transforme les pixels de l'image en deux valeurs : les pixels ayant une intensité supérieure à 150 deviennent blancs (255), tandis que ceux ayant une intensité inférieure ou égale à 150 deviennent noirs (0). Cela permet de créer une image binaire où les objets d'intérêt sont mis en évidence par rapport à l'arrière-plan. 

_,th = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

cv.namedWindow('image seuillée', cv.WINDOW_NORMAL)
cv.imshow('image seuillée', np.uint8(th))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 5 : Espaces de couleur ===
# ------------------------------------
# Conversion de l’image filtrée en espace HSV
# TODO : effectuez la conversion avec cv.cvtColor.

hsv = cv.cvtColor(filtered, cv.COLOR_BGR2HSV)

# Calcul et affichage des histogrammes des trois canaux
colors = ('r', 'g', 'b')
labels = ('h', 's', 'v')

for i, col in enumerate(colors):
    # TODO : calculez l’histogramme du canal i
    hist = cv.calcHist([hsv], [i], None, [256], [0, 256])
    plt.plot(hist, color=col, label=labels[i])
    plt.xlim([0, 256])

plt.legend()
plt.show()
plt.savefig('fig1.pdf')

# QUESTION : Que représente la courbe sauvegardée dans fig1.pdf ?
# la courbe représente l'histogramme des valeurs de teinte (H), de saturation (S) et de valeur (V) de l'image dans l'espace de couleur HSV. 

# Quelle information donne-t-elle sur l’image ?
# elle donne des informations sur la distribution des couleurs dans l'image, montrant quelles teintes sont les plus présentes, ainsi que le niveau de saturation et de luminosité des pixels.


# === Étape 6 : Extraction et affichage des canaux ===
# ----------------------------------------------------
# TODO : extrayez les canaux H et S à partir de hsv.
h = hsv[:,:,0]
s = hsv[:,:,1]

cv.namedWindow('teinte (Hue)', cv.WINDOW_NORMAL)
cv.imshow('teinte (Hue)', np.uint8(h))
cv.waitKey(0)
cv.destroyAllWindows()

cv.namedWindow('saturation', cv.WINDOW_NORMAL)
cv.imshow('saturation', np.uint8(s))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 7 : Seuillage sur teinte et saturation ===
# ----------------------------------------------------
# TODO : Créez un masque pour isoler certaines couleurs.
# Exemple : détection de tons rouges à forte saturation.
thh = cv.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv.inRange(hsv, (160, 100, 100), (180, 255, 255))

cv.namedWindow('image seuillée H et S', cv.WINDOW_NORMAL)
cv.imshow('image seuillée H et S', np.uint8(thh))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 8 : Morphologie (érosion / ouverture) ===
# ---------------------------------------------------
# Création de l’élément structurant
# TODO : essayez différentes formes : RECT, ELLIPSE, CROSS
erosion_shape = cv.MORPH_CROSS
erosion_size = 1

element = cv.getStructuringElement(
    erosion_shape,
    (2 * erosion_size + 1, 2 * erosion_size + 1),
    (erosion_size, erosion_size)
)

# TODO : appliquez une opération morphologique (érosion ou ouverture)
erosion_dst = cv.erode(thh, element)

cv.namedWindow('image morphologique', cv.WINDOW_NORMAL)
cv.imshow('image morphologique', np.uint8(erosion_dst))
cv.waitKey(0)
cv.destroyAllWindows()


# === Étape 9 : Composantes connexes ===
# --------------------------------------
# TODO : appliquez cv.connectedComponents() à l’image binaire.
retval, labels = cv.connectedComponents(erosion_dst)

print("Nombre de composantes détectées :", retval)

# Parcours des composantes détectées

for i in range(retval):
    idcc = np.where(labels == i)
    if len(idcc[0]) > 1000:  # seuil minimal de taille
        cv.namedWindow('composante', cv.WINDOW_NORMAL)
        cv.imshow('composante', np.uint8((labels == i) * 255))

        # TODO : calculez et affichez la moyenne des coordonnées.
        print("Moyenne des coordonnées de la composante", i, ":", np.mean(idcc, axis=1))

        cv.waitKey(500)
        cv.destroyAllWindows()

# Exercice supplémentaire pour ceux qui ont fini.
# Afficher la photo des tulipes avec une petite croix au milieu 
# de chaque tulipe détectée et le nombre de tulipes en bas.