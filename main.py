from utils import *

pathImg = "Resources/1.jpg"
heightImg = 450
widthImg = 450



#### 1. PREPARE THE IMAGE
# img = cv2.imread("Resources/1.jpg")
img = cv2.imread(pathImg)
img = cv2.resize(img, (widthImg,heightImg)) #RESIZE IMAGE TO MAKE IT SQAURE
imgBlank = np.zeros((heightImg,widthImg,3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING, DEBUGGING
imgThreshold = preProcess(img)

#### 2. FIND ALL CONTOURS
imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSE
imgBiggestContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSE
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # DRAW ALL DETECTED CONTOURS

#### 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
biggest, max_area = biggestContour(contours)  # FIND THE BIGGEST CONTOUR


imageArray = ([img,imgThreshold,imgContours,imgBlank],
              [imgBlank,imgBlank,imgBlank,imgBlank])
stackedImage = stackImages(imageArray,1)
cv2.imshow('Stacked Images',stackedImage)

cv2.waitKey(0)