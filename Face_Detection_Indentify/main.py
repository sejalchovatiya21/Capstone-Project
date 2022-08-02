# import libraries of python OpenCV
import cv2
import os

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

# capture frames from a camera
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
# loop runs if capturing has been initialized.
while 1:

	# reads frames from a camera
	ret, img = cam.read()

	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		count += 1

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Save the captured image into the datasets folder
		cv2.imwrite("dataset_face/User." + str(face_id) + '.' + str(count) + ".jpg", img[y:y + h, x:x + w])

		# Detect eyes of different sizes in the input image
		eyes = eye_cascade.detectMultiScale(roi_gray)

		# To draw a rectangle in eyes
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

		# Save the captured image into the datasets folder
		cv2.imwrite("dataset_eye/User." + str(face_id) + '.' + str(count) + ".jpg", img[y:y + h, x:x + w])

	# Display an image in a window
	cv2.imshow('img', img)

	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 113:
		break
	elif count >= 15:  # Take 30 face sample and stop video
		break

# Close the window
cam.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
