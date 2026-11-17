import numpy as np
import matplotlib
matplotlib.use('agg') # A SUPPRIMER EN FONCTION DE L AFFICHAGE
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import linear_model

imglinit = cv.imread("lignes.jpg")

image_base = cv.resize(imglinit, (512, 512))

# traitement description de l'image : conversion et laplacien apres flou gaussien 
imglblr = cv.GaussianBlur(image_base, (55, 55), 0)

ddepth = cv.CV_16S
src_gray = cv.cvtColor(imglblr, cv.COLOR_BGR2GRAY)
imgl = cv.Laplacian(src_gray, ddepth, ksize=3)
imgl = cv.convertScaleAbs(imgl)

# normalisation et seuillage
imgl = imgl.astype(float)
imgl = 255 * ((imgl / np.max(imgl)) > 0.75)

# affichage des points d'interet detectes
cv.imshow("img", np.uint8(imgl))
cv.waitKey(0)
cv.destroyAllWindows()

x, y = np.where(imgl==255)
xx = x**2
var1 = np.ones_like(x)
X = np.stack((xx, x, var1), axis=1)

ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Calcul de la courbe par prédiction RANSAC
line_X = np.arange(x.min(), x.max()) # X
line_XX = line_X**2                  # X^2
line_1 = np.ones_like(line_X)        # 1
llll = np.stack((line_XX, line_X, line_1), axis=1)
line_y_ransac = ransac.predict(llll) # Y = a*X^2 + b*X + c*1

plt.close('all')
plt.imshow(np.uint8(image_base))
lw = 2
plt.scatter(
    y[inlier_mask], x[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    y[outlier_mask], x[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(
    line_y_ransac,
    line_X,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.show()
plt.savefig('fig2.pdf')

poly_a = ransac.estimator_.coef_[0]
poly_b = ransac.estimator_.coef_[1]
poly_c = ransac.estimator_.intercept_

print(f'coefficients du polynome : a={poly_a}, b={poly_b}, c={poly_c}.')