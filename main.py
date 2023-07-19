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

# cv2.imshow("Image", img)
# cv2.waitKey(0)



imageArray = ([img,imgBlank,imgBlank,imgBlank],
              [imgBlank,imgBlank,imgBlank,imgBlank])
stackedImage = stackImages(imageArray,1)
cv2.imshow('Stacked Images',stackedImage)

cv2.waitKey(0)