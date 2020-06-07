import cv2
import imStitchFunctions as isf

imgA = cv2.imread('IMG_3595.jpg')
imgB = cv2.imread('IMG_3596.jpg')
imgC = cv2.imread('IMG_3597.jpg')
imgD = cv2.imread('IMG_3598.jpg')

#adding images to list
images = []
images.append(imgA)
images.append(imgB)
images.append(imgC)
images.append(imgD)

#now stitching them together
stitched = isf.stitchImages(images)

cv2.imshow("stitched",stitched)
cv2.waitKey(0)

cv2.imwrite("panorama.jpg", stitched)
