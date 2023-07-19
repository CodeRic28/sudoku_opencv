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
imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSE
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # DRAW ALL DETECTED CONTOURS

#### 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
biggest, max_area = biggestContour(contours)  # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0,0,255),20)  # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    # DISPLAY THE SUDOKU USING CONTOUR POINTS OF BIGGEST CONTOUR
    imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)



imageArray = ([img,imgThreshold,imgContours,imgBigContour],
              [imgWarpColored,imgBlank,imgBlank,imgBlank])
stackedImage = stackImages(imageArray,1)
cv2.imshow('Stacked Images',stackedImage)

cv2.waitKey(0)