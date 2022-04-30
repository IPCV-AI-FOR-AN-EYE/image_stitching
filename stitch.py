from features import *
import cv2

numberofimages = 3
im1 = cv2.imread('1.jpeg')
im2 = cv2.imread('2.jpeg')
im3 = cv2.imread('3.jpeg')

images = [im1, im2, im3]

if numberofimages == 2:
    (finalimage, pointsimage) = stitch2images([images[0], images[1]], matched=True)
else:
    (finalimage, pointsimage) = stitch2images([images[numberofimages-2], images[numberofimages-1]], matched=True)
    for i in range(numberofimages - 2):
        (finalimage, pointsimage) = stitch2images([images[numberofimages-i-3],finalimage], matched=True)

cv2.imwrite("pointsimage.jpg", pointsimage)
cv2.imwrite("finalimage.jpg", finalimage[:,:int(finalimage.shape[1]*0.9)])



