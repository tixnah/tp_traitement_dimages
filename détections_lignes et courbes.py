# TODO. Étudier ce code en affichant les images intermédiaires 
# et les vecteurs produits. 
# Ensuite, remplacer les points d'intérets calculés par Sobel 
# avec des points provenants de la transformation de hough 
# (voir fichier hough.py fourni)
# Enfin, vous pourrez vous baser sur courbes.py pour approcher
# les courbes par des polynomes de degré 2. 

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

imglinit = cv.imread("ligne2.jpg")
# mise à des dimensions fixes
image_base = cv.resize(imglinit, (512, 512))

# Flou gaussien et conversion en niveaux de gris
imglblr = cv.GaussianBlur(image_base, (55, 55), 0)
src_gray = cv.cvtColor(imglblr, cv.COLOR_BGR2GRAY)

# Calcul du Laplacien et normalisation
ddepth = cv.CV_16S
imgl = cv.Laplacian(src_gray, ddepth, ksize=3)
imgl = cv.convertScaleAbs(imgl)

# Normalisation et seuillage
imgl = imgl.astype(float)
imgl = 255 * (imgl > 7.5)

imgl = np.uint8(imgl)
y, x = np.where(imgl == 255)

print("Nombre de points détectés :", len(x))

# Calcul des gradients
grad_x = cv.Sobel(src_gray, cv.CV_32F, 1, 0, ksize=3)
grad_y = cv.Sobel(src_gray, cv.CV_32F, 0, 1, ksize=3)

# Extraction des gradients aux points détectés
gx = grad_x[y, x]
gy = grad_y[y, x]

# Orientation locale du gradient
angles = np.arctan2(gy, gx)

# Tableau des points d'intérêt :
# colonne 0 : x
# colonne 1 : y
# colonne 2 : angle

points_interet = np.column_stack((x, y, angles))

print("\nTableau des points d'intérêt :")
print("[x, y, angle en radians]")
print(points_interet)

theta_cos = np.cos(2 * angles)
theta_sin = np.sin(2 * angles)

# Nouveau tableau de caractéristiques
features = np.column_stack((x, y, theta_cos, theta_sin))
scaler = StandardScaler()
features_normalisees = scaler.fit_transform(features)


clustering = DBSCAN(
    eps=0.55,
    min_samples=5
)

labels = clustering.fit_predict(features_normalisees)

# Suppression des clusters de moins de 10 points

taille_min = 10

for k in set(labels):
    if k == -1:
        continue

    masque = labels == k

    if np.sum(masque) < taille_min:
        labels[masque] = -1

# Recalcul du nombre de clusters conservés
clusters_conserves = set(labels) - {-1}

n_clusters = len(clusters_conserves)

n_bruit = np.sum(labels == -1)

print("Nombre de clusters conservés :", n_clusters)
print("Nombre de points de bruit :", n_bruit)

for k in sorted(clusters_conserves):
    print(f"Cluster {k} : {np.sum(labels == k)} points")


# Affichage
plt.figure(figsize=(12, 12))
plt.imshow(
    cv.cvtColor(image_base, cv.COLOR_BGR2RGB)
)
longueur_vecteur = 18
couleurs = plt.cm.tab20(
    np.linspace(0, 1, max(n_clusters, 1))
)
for k in range(n_clusters):
    masque = labels == k
    x_cluster = x[masque]
    y_cluster = y[masque]
    theta_cluster = angles[masque]
    couleur = couleurs[k]
    plt.scatter(
        x_cluster,
        y_cluster,
        color=couleur,
        s=12,
        label=f"Cluster {k}"
    )
    v = longueur_vecteur * np.cos(theta_cluster)
    u = -longueur_vecteur * np.sin(theta_cluster)
    plt.quiver(
        x_cluster,
        y_cluster,
        u,
        v,
        color=couleur,
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.003
    )

# Affichage du bruit
masque_bruit = labels == -1

plt.scatter(
    x[masque_bruit],
    y[masque_bruit],
    color="white",
    marker="x",
    s=12,
    label="Bruit"
)

# Paramètres d'affichage
plt.title("Points d'intérêt, clusters et orientations locales")
plt.legend()
plt.axis("off")
plt.tight_layout()
plt.show()
