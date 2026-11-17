import cv2 as cv
import numpy as np
import math

img = cv.imread(cv.samples.findFile('ligne2.jpg'))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=5)
lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=3, maxLineGap=3)

for line in lines:
    x1, y1, x2, y2 = line[0]

    angle = math.degrees(math.atan2(y2-y1, x2-x1))
    angle = angle % 180

    print(f"Angle : {angle:.2f} degrés")

    cv.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv.putText(img, f"{angle:.1f}°", (x1,y1),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

cv.imshow("Lignes et angles", img)
cv.imwrite('houghlines5.jpg', img)

cv.waitKey(0)
cv.destroyAllWindows()
