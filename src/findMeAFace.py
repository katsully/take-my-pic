import cv2
from keras.models import load_model
import numpy as np
import dlib
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client
import subprocess
import time
from imutils import face_utils
import imutils
import random
import re

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from utils.kats_helper import landmarks_to_np
from utils.kats_helper import get_centers
from utils.kats_helper import get_aligned_face
from utils.kats_helper import judge_eyeglass
from utils.kats_helper import get_color_name
from utils.kats_helper import rgb_to_hsv
from utils.kats_helper import hsv_to_rgb
from utils.kats_helper import ColorNames

from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3,960)	# width
cam.set(4,720)  # height
width = 960
height = 720
face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# eyeglasses
predictor_path = "../data/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# build udp_client for osc protocol
client = udp_client.UDPClient("", 8001)
# counter for time with no faces
noface_counter = 0
# counter for collecting the avg info about a person
avg_counter = 0
face_counter = 0
emotion_text = []
# gender_text = []
wearing_glasses = []
shirt_color = []
found_face = False;
face_x = 0
face_y = 0
face_w = 0
face_h = 0
face_analyze = False
face_looking = False
face_loop = 0
find_face_mode = False

# hyper-parameters for bounding boxes shape
crop_offsets = (50, 70)
emotion_offsets = (20, 40)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
# gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# OSC
client = udp_client.UDPClient("10.18.235.227", 8001)

def sendCoords(x,y,w,h):
	rect = dlib.rectangle(x,y,x+w,y+h)
	landmarks = predictor(gray_img, rect)
	landmarks = landmarks_to_np(landmarks)
	inner_eye = landmarks[3]
	cv2.circle(img, (inner_eye[0], inner_eye[1]), 20, 100)
	msg = osc_message_builder.OscMessageBuilder(address="/lookAt")

	xPos = inner_eye[0]/cam.get(cv2.CAP_PROP_FRAME_WIDTH)
	yPos = inner_eye[1]/cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
	msg.add_arg(xPos)
	msg.add_arg(yPos)
	msg = msg.build()
	client.send(msg)


if cam.isOpened(): # try to get the first frame
	ret, img = cam.read()
	print("width ", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
else:
    ret = False

while(ret):
	ret, img = cam.read()
	# flip camera 90 degrees
	# rotate = imutils.rotate_bound(img, 90)
	# flipped = cv2.flip(rotate, 1)
	# convert image to grayscale
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# convert from bgr to rgb
	rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# gray is the input grayscale image
	# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
	# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
	# minSize (optional) is the minimum rectangle size to be considered a face
	faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
	# if no faces are detected
	# when faces are detected, the faces variable is an array of tuple, when no faces are detected the faces variable is an empty tuple
	if isinstance(faces, tuple):
		noface_counter+=1
		found_face = False
		face_analyze = False
		face_counter = 0
		# reset average face counter
		avg_counter = 0
		emotion_text.clear()
		# gender_text.clear()
		wearing_glasses.clear()
		shirt_color.clear()
		# only send message if no faces detected after 10 seconds
		if(noface_counter > 200):
			msg = osc_message_builder.OscMessageBuilder(address="/noFace")
			msg = msg.build()
			client.send(msg)
			noface_counter = 0

	# camera found one or more faces
	else:
		# reset noFaces timer
		noface_counter = 0
		# focusing on a single face
		if found_face:
			x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
			# crop image so we only focus on this face
			cropped_img = gray_img[y1:y2, x1:x2]
			faces = face_detector.detectMultiScale(cropped_img, scaleFactor=1.3, minNeighbors=5)
			# is the face gone?
			if isinstance(faces, tuple):
				# tell matt we lost the face after a significant amount of tracking
				if face_analyze:
					msg = osc_message_builder.OscMessageBuilder(address="/lostFace")
					msg = msg.build()
					client.send(msg)
				found_face = False
				face_counter = 0
				face_analyze = False
				emotion_text.clear()
				shirt_color.clear()
				wearing_glasses.clear()
				# print("lost your face!")
			# face is still there
			else:
				# send coordinates to max so avatar looks at face
				sendCoords(x,y,w,h)
				# rect = dlib.rectangle(x,y,x+w,y+h)
				# landmarks = predictor(gray_img, rect)
				# landmarks = landmarks_to_np(landmarks)
				# inner_eye = landmarks[3]
				# cv2.circle(img, (inner_eye[0], inner_eye[1]), 20, 100)
				# msg = osc_message_builder.OscMessageBuilder(address="/lookAt")
				# # print("xPos ", inner_eye[0])

				# xPos = inner_eye[0]/cam.get(cv2.CAP_PROP_FRAME_WIDTH)
				# yPos = inner_eye[1]/cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
				# msg.add_arg(xPos)
				# msg.add_arg(yPos)
				# msg = msg.build()
				# client.send(msg)

				# if we're still determining this face isn't someone quickly entering and exiting
				if not face_analyze:
					face_counter += 1
					# face isn't just coming and going
					if face_counter > 25:
						face_analyze = True
				# we're ready to analyze this face!
				if face_analyze:
					if faces.size > 1 or random.random() <= 0.3:
						# print("analyzing face!")
						x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), emotion_offsets)
						gray_face_og = gray_img[y1:y2, x1:x2]
						x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
						rgb_face_og = rgb_img[y1:y2, x1:x2]
						try:
							gray_face = cv2.resize(gray_face_og, (emotion_target_size))
							# rgb_face = cv2.resize(rgb_face_og, (gender_target_size))
						except:
							continue
						# get emotion
						gray_face = preprocess_input(gray_face, False)
						gray_face = np.expand_dims(gray_face, 0)
						gray_face = np.expand_dims(gray_face, -1)
						emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
						emotion_text.append(emotion_labels[emotion_label_arg])

						# eyeglasses
						rect = dlib.rectangle(x,y,x2,y2)
						landmarks = predictor(gray_img, rect)
						landmarks = landmarks_to_np(landmarks)
						LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(img, landmarks)
						aligned_face = get_aligned_face(gray_img, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
						wearing_glasses.append(judge_eyeglass(aligned_face))

						avg_counter += 1

						# shirt color
						s_y = face_y + int(face_h * 1.65)
						s_h = face_h + int(face_h * 1.25)
						s_y2 = s_y+s_h
						# ensure shirt region doesn't extend outside of image
						if x2 >= cam.get(cv2.CAP_PROP_FRAME_WIDTH):
							x2 = cam.get(cv2.CAP_PROP_FRAME_WIDTH) -1
						if s_y2 >= cam.get(cv2.CAP_PROP_FRAME_HEIGHT):
							s_y2 = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) -1
						if x1 < 0:
							x1 = 0
						shirt_region = rgb_img[s_y:int(s_y2), x1:x2]

						if type(shirt_region) is not 'NoneType':
							shirt_image = Image.fromarray(shirt_region, 'RGB')
							hist = shirt_image.histogram()
							# split into red, green, blue
							r = hist[0:256]
							g = hist[256:256*2]
							b = hist[256*2: 256*3]

							# perform the weighted average of each channel
							# the *index* is the channel value, and the *value* is its weight
							avg_r = sum( i*w for i, w in enumerate(r) ) / sum(r)
							avg_g = sum( i*w for i, w in enumerate(g) ) / sum(g)
							avg_b = sum( i*w for i, w in enumerate(b) ) / sum(b)

							avg_h, avg_s, avg_v = rgb_to_hsv(avg_r, avg_g, avg_b)
							# print("rgb", avg_r, avg_g, avg_b)
							# up the saturation
							avg_s += .12
							avg_v += avg_v * 1.75

							new_r, new_g, new_b = hsv_to_rgb(avg_h, avg_s, avg_v)
							# print("new averages", new_r, new_g, new_b)
							cv2.rectangle(img,(face_x,s_y), (face_x+face_w, s_y+face_h), (0,255,0), 2)
							# actual_name, closest_name = get_color_name( (int(new_r),int(new_g),int(new_b)) )
							color_name = ColorNames.findNearestImageMagickColorName((int(new_r),int(new_g),int(new_b)))
							# magik_name = ColorNames.findNearestImageMagickColorName((int(new_r),int(new_g),int(new_b)))
							# print(color_name)
							shirt_color.append(color_name)

						if avg_counter > 50:
							msg = osc_message_builder.OscMessageBuilder(address="/takeAPic")
							msg = msg.build()
							client.send(msg)
							# get timestamp
							ts = time.gmtime()
							timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
							fileName = "../faces/face" + timestamp + ".jpg"
							cv2.imwrite(fileName, rgb_face_og)
							avg_shirt_color = max(set(shirt_color), key=shirt_color.count)
							shirt_list = re.findall('[A-Z][^A-Z]*', avg_shirt_color)
							if shirt_list != []:
								shirt_list = " ".join(shirt_list)
							else:
								shirt_list = avg_shirt_color
							shirt_num = ""
							shirt_addon = ""
							shirt_addon2 = ""
							if re.findall('\d+', shirt_list) != []:
								shirt_num = re.findall('\d+', shirt_list)[0]
								shirt_list = ''.join([i for i in shirt_list if not i.isdigit()])
								last_digit = int(shirt_num) % 10
								if last_digit == 2:
									shirt_addon2 = " second"
								elif last_digit == 3:
									shirt_addon2 = " third"
								elif last_digit == 4:
									shirt_addon2 = " fourth"
								elif last_digit == 5:
									shirt_addon2 = " fifth"
								elif last_digit == 6:
									shirt_addon2 = " sixth"
								elif last_digit == 7:
									shirt_addon2 = " seventh"
								elif last_digit == 8:
									shirt_addon2 = " eighth"
								elif last_digit == 9:
									shirt_addon2 = " nineth"
								shirt_addon = "in my" + shirt_addon2 + " favorite shade of "
							shirt_caption = shirt_list + " top " + shirt_addon + shirt_list
							# sad, surprise, happy, angry, neutral, disgust, and fear
							emotion_caption = max(set(emotion_text), key=emotion_text.count)
							if emotion_caption == "sad":
								emotion_caption = "I want to remember you with the forlorn face wearing a "
							elif emotion_caption == "happy":
								emotion_caption = "I remember a bright face sporting a "
							elif emotion_caption == "neutral":
								emotion_caption = "They were a bore and everyone knew it. They always wore a"
							else:
								emotion_caption += " person wearing a "
							caption = emotion_caption + shirt_caption 
							if max(set(wearing_glasses), key=wearing_glasses.count):
								caption += " and bespeckled eyes"
							subprocess.call([r'C:/Users/student/Documents/GabeSanFran/take-my-pic/insta.bat', fileName, caption])
							# exit the loop
							# ret = False
							# Reset everything
							avg_counter = 0
							emotion_text.clear()
							# gender_text.clear()
							wearing_glasses.clear()
							shirt_color.clear()
							found_face = False
							face_analyze = False
							face_counter = 0
							find_face_mode = False
				
		# still looking for a face to focus on
		else:	
			# if more than one face, loop through looking at face
			if faces.size > 1 and face_looking is False and find_face_mode is False:
				face_looking = True
				print("LOOKING MODE")
			np.random.shuffle(faces)
			for (x,y,w,h) in faces: 
				if face_looking:
					# look at each frame for three seconds
					t_end = time.time() + 3
					msg = osc_message_builder.OscMessageBuilder(address="/newFace")
					msg = msg.build()
					client.send(msg)
					while time.time() < t_end:
						sendCoords(x,y,w,h)
				# we are now focusing on one face
				else:
					# print("FINDING FACE")
					found_face = True;
					face_x, face_y, face_w, face_h = x,y,w,h
					break

			face_loop += 1
			if face_loop > 3:
				face_loop = 0
				face_looking = False
				find_face_mode = True

	cv2.imshow('test window', img)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break
 
# # end of program
cam.release()
cv2.destroyAllWindows()
