import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

frame = cv2.imread('IMG_3336.jpg')
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)

center = np.zeros((len(ids), 2))
for i in range(len(ids)):
    f = corners[i][0]
    center[i, 0] = np.mean(f[:, 0])
    center[i, 1] = np.mean(f[:, 1])

corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=parameters)
img2 = aruco.drawDetectedMarkers(frame, corners, ids)

cv2.imshow('image', img2)
cv2.waitKey(0)


