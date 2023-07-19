import cv2
import numpy as np


#### 1 - PREPROCESSING IMAGE
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to Grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Add Gaussian Blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # Add Adaptive Threshold
    return imgThreshold

#### 3. FINDING THE BIGGEST CONTOUR ENSURING ITS A SUDOKU PUZZLE
def biggestContour(contours):
    biggest = np.array([])  # WILL CONTAIN ALL THE CORNER POINTS, NOT NECESSARILY ORDERED
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)  # FIND PERIMETER
            approx = cv2.approxPolyDP(i, 0.02*peri,True)  # FIND THE NUMBER OF CORNERS
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

#### 3. REORDER POINTS FOR WARP PERSPECTIVE
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

#### 6 - TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        # CREATING HORIZONTAL STACKS FOR EACH ROW
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver