import cv2

while 1:

    myImage = cv2.imread('image_01.jpg', 1)
    print(myImage)

    cv2.imshow('mythumbnail', myImage)
    cv2.waitKey(200)

    myImage = cv2.imread('image_01.jpg', 0)
    print(myImage)

    cv2.imshow('mythumbnail', myImage)
    cv2.waitKey(200)

cv2.destroyAllWindows()

import cv2
import numpy as np

# load image and get dimensions
img = cv2.imread("image_01.jpg")
h, w, c = img.shape

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# do floodfill
result = img.copy()
cv2.floodFill(result, mask, (0,0), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (38,313), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (363,345), (255,255,255), (3,151,65), (3,151,65), flags=8)
cv2.floodFill(result, mask, (619,342), (255,255,255), (3,151,65), (3,151,65), flags=8)

# write result to disk
cv2.imwrite("image_01.jpg", result)

# display it
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()