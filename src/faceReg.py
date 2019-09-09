import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3,640)	# width
cam.set(4,480)  # height
face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')

# For each person, enter one numeric id
face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look at the camera and wait.. ")

# Initializing individual sampling face count
while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray is the input grayscale image
	# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
	# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
	# minSize (optional) is the minimum rectangle size to be considered a face
	faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0),2)
		count += 1
		# Save the captured image into the datasets folder
		cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
		cv2.imshow('image', img)


	k = cv2.waitKey(30) & 0xff
	if k == 27: 	# press ESC to quit
		break
	elif count >= 30: # Take 30 face samples and stop video

# end of program
print("\n [INFO] Exiting Program")
cap.release()
cv2.destroyAllWindows()