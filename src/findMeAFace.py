import cv2
from keras.models import load_model
import numpy as np
import dlib
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client
import subprocess
import time
import imutils
from statistics import mode

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from utils.glasses_helper import landmarks_to_np
from utils.glasses_helper import get_centers
from utils.glasses_helper import get_aligned_face
from utils.glasses_helper import judge_eyeglass

cam = cv2.VideoCapture(0)
cam.set(3,640)	# width
cam.set(4,480)  # height
face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')

# eyeglasses
predictor_path = "../data/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# build udp_client for osc protocol
client = udp_client.UDPClient("127.0.0.1", 8001)
# counter for time with no faces
counter = 0
# counter for collecting the avg info about a person
avg_counter = 0
emotion_text = []
gender_text = []
wearing_glasses = []

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]


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
	gray_img = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
	# convert from bgr to rgb
	rgb_img = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
	# gray is the input grayscale image
	# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
	# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
	# minSize (optional) is the minimum rectangle size to be considered a face
	faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
	# if no faces are detected
	# when faces are detected, the faces variable is an array of tuple, when no faces are detected the faces variable is an empty tuple
	if isinstance(faces, tuple):
		counter+=1
		# reset average face counter
		avg_counter = 0
		emotion_text = []
		gender_text = []
		wearing_glasses = []
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
			x1,x2,y1,y2 = apply_offsets((x,y,w,h), emotion_offsets)
			gray_face = gray_img[y1:y2, x1:x2]
			x1,x2,y1,y2 = apply_offsets((x,y,w,h), gender_offsets)
			rgb_face_og = rgb_img[y1:y2, x1:x2]
			try:
				gray_face = cv2.resize(gray_face, (emotion_target_size))
				rgb_face = cv2.resize(rgb_face_og, (gender_target_size))
			except:
				continue
			# get emotion
			gray_face = preprocess_input(gray_face, False)
			gray_face = np.expand_dims(gray_face, 0)
			gray_face = np.expand_dims(gray_face, -1)
			emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
			emotion_text.append(emotion_labels[emotion_label_arg])

			# get gender
			rgb_face = np.expand_dims(rgb_face, 0)
			rgb_face = preprocess_input(rgb_face, False)
			gender_prediction = gender_classifier.predict(rgb_face)
			gender_label_arg = np.argmax(gender_prediction)
			gender_text.append(gender_labels[gender_label_arg])

			# eyeglasses
			rect = dlib.rectangle(x,y,x+w,y+h)
			landmarks = predictor(gray_img, rect)
			landmarks = landmarks_to_np(landmarks)
			LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(flipped, landmarks)
			aligned_face = get_aligned_face(gray_img, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
			wearing_glasses.append(judge_eyeglass(aligned_face))

			avg_counter += 1
			print (avg_counter)

			if avg_counter > 50:
				print("HERE")
				# get timestamp
				ts = time.gmtime()
				timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
				fileName = "../faces/face" + timestamp + ".jpg"
				cv2.imwrite(fileName, rgb_face_og)
				caption = mode(emotion_text) + " " + mode(gender_text)
				if mode(wearing_glasses):
					caption += " and is wearing glasses"
				subprocess.call([r'C:/Users/NUC6-USER/take-my-pic/insta.bat', fileName, caption])
				# exit the loop
				ret = False
				avg_counter = 0
				emotion_text = []
				gender_text = []
				wearing_glasses = []

	cv2.imshow('test window', flipped)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break
 
# end of program
cam.release()
cv2.destroyAllWindows()
