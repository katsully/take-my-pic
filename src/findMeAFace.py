import cv2
from threading import Timer
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client
import subprocess
import time
import imutils

cam = cv2.VideoCapture(0)
cam.set(3,640)	# width
cam.set(4,480)  # height
face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')

# build udp_client for osc protocol
client = udp_client.UDPClient("127.0.0.1", 8001)
counter = 0


if cam.isOpened(): # try to get the first frame
	ret, img = cam.read()
else:
    ret = False

while(ret):
	ret, img = cam.read()
	# flip camera 90 degrees
	rotate = imutils.rotate_bound(img, 90)
	flipped = cv2.flip(rotate, 1)
	# convert image to grayscale
	gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
	# gray is the input grayscale image
	# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
	# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
	# minSize (optional) is the minimum rectangle size to be considered a face
	faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	# if no faces are detected
	# when faces are detected, the faces variable is an array of tuple, when no faces are detected the faces variable is an empty tuple
	if isinstance(faces, tuple):
		counter+=1
		# only send message if no faces detected after 10 seconds
		if(counter > 200):
			msg = osc_message_builder.OscMessageBuilder(address="/noFace")
			msg = msg.build()
			client.send(msg)
			counter = 0

	# camera found one or more faces
	else:
		# reset noFaces timer
		counter = 0
		for (x,y,w,h) in faces:
			# get emotion
			# emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
			# emotion_text = emotion_labels[emotion_label_arg]

			# get timestamp
			ts = time.gmtime()
			# timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
			# fileName = "../faces/face" + timestamp + ".jpg"
			# cv2.imwrite(fileName, img[y:y+h, x:x+w])
			# subprocess.call([r'C:/Users/NUC6-USER/take-my-pic/insta.bat', fileName])
			# # exit the loop
			# ret = False

	cv2.imshow('test window', flipped)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break
 
# end of program
cam.release()
cv2.destroyAllWindows()
