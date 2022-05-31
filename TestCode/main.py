import cv2

myImage = cv2.imread('image_02.jpg', 1)
print(myImage)

cv2.imshow('mythumbnail', myImage)
cv2.waitKey(5000)
cv2.destroyAllWindows()