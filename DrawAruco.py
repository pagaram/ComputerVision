import cv2
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

name = 'image'
file = '.png'
for i in range(50):
    img = aruco.drawMarker(aruco_dict, i, 700)
    filename = name + str(i) + file
    cv2.imwrite(filename, img)
    print("image" + str(i))








