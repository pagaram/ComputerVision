import numpy as np
import cv2
import imutils

def createMask(stitched):
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)) # making 10 pixel border
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1] #thresholding inside border
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) #creating mask

    return stitched, mask, thresh

def cropPanorama(stitched, mask, thresh):
    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c) #compute bounding area of panorama

    stitched = stitched[y:y + h, x:x + w] #cropped stitched image

    return stitched

def stitchImages(images):
    stitcher = cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images) #using opencv stitch function

    stitched, mask, thresh = createMask(stitched) #creating mask for bordered area

    stitched = cropPanorama(stitched, mask, thresh) #determining bounding box and cropping image

    return stitched
