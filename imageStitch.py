import numpy as np
import cv2
import matplotlib.pyplot as plt

imgA = cv2.imread('IMG_3595.jpg')
img1 = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)

imgB = cv2.imread('IMG_3597.jpg')
img2 = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.2

orb = cv2.ORB_create(MAX_FEATURES)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

imgA2 = np.zeros(shape=[img1.shape[0], img1.shape[1], 3], dtype=np.uint8)
cv2.drawKeypoints(imgA, kp1, imgA2, color=(0, 255, 0), flags=0)

imgB2 = np.zeros(shape=[img2.shape[0], img2.shape[1], 3], dtype=np.uint8)
cv2.drawKeypoints(imgB, kp2, imgB2, color=(0, 255, 0), flags=0)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches = matcher.match(des1, des2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

imMatches = cv2.drawMatches(imgA, kp1, imgB, kp2, matches, None)

plt.figure()
plt.imshow(imMatches)
plt.show()

pointsA = np.zeros((len(matches), 2), dtype=np.float32)
pointsB = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    pointsA[i, :] = kp1[match.queryIdx].pt
    pointsB[i, :] = kp2[match.trainIdx].pt

#compute homography
h, mask = cv2.findHomography(pointsB, pointsA, cv2.RANSAC, 5.0)

print(h)

height, width = img1.shape
pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, h)

img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

plt.figure()
plt.imshow(img2)
plt.show()

dst = cv2.warpPerspective(imgA, h, (imgB.shape[1] + imgA.shape[1], imgB.shape[0]))
dst[0:imgB.shape[0], 0:imgB.shape[1]] = imgA

plt.figure()
plt.imshow(dst)
plt.show()









