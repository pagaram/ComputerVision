import cv2
import numpy as np

img = cv2.imread('coins.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 100)
circles = np.uint16(np.around(circles))

if circles is not None:
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 4)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 1, (0, 0, 255), 3)

cv2.imshow('detected circles', img)
cv2.waitKey(0)
