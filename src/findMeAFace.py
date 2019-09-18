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
from utils.kats_helper import rgb_to_hsv
from utils.kats_helper import hsv_to_rgb
from utils.kats_helper import ColorNames
from utils.kats_helper import resizeAndPad

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


cam = cv2.VideoCapture(0)
cam.set(3,1920)	# width
cam.set(4,1080)  # height
face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')

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
gender_text = []
shirt_brightness = ""

cropped_img = None

# hyper-parameters for bounding boxes shape
crop_offsets = (50, 70)
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# OSC
client = udp_client.UDPClient("10.18.235.227", 8001)

looking_for = ["glasses", "man", "woman", "light_shirt", "dark_shirt"]
looking_for_count = 0
num_of_pics = 0
test_pass = False

# font
ft_bold = ImageFont.truetype("NewsGothicStd-BoldOblique.otf",35)
ft_color = ImageFont.truetype("News Gothic Regular.otf",30)
ft_collection = ImageFont.truetype("News Gothic Regular.otf",15)

# captions
text_file = open("emotions.txt", "r")
emotion_list = [line.rstrip() for line in text_file.readlines()]
emotion_list_counter = 0
angry_file = open("anger.txt", "r")
sad_file = open("sadness.txt", "r")
disgust_file = open("disgust.txt", "r")
happy_file = open("happiness.txt", "r")
surprise_file = open("surprise.txt", "r")
neutral_file = open("neutral.txt", "r")
fear_file = open("fear.txt", "r")
angry_list = [line.rstrip() for line in angry_file.readlines()]
angry_file.close()
sad_list = [line.rstrip() for line in sad_file.readlines()]
sad_file.close()
disgust_list = [line.rstrip() for line in disgust_file.readlines()]
disgust_file.close()
happy_list = [line.rstrip() for line in happy_file.readlines()]
happy_file.close()
surprise_list = [line.rstrip() for line in surprise_file.readlines()]
surprise_file.close()
neutral_list = [line.rstrip() for line in neutral_file.readlines()]
neutral_file.close()
fear_list = [line.rstrip() for line in fear_file.readlines()]
fear_file.close()
angry_counter = 0
sad_counter = 0
disgust_counter = 0
happy_counter = 0
surprise_counter = 0
neutral_counter = 0
fear_counter = 0
new_emotion=""

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

if cam.isOpened(): # try to get the first frame
	ret, img = cam.read()
	# print("width ", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	# print("height ", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    ret = False

while(ret):
	# moments enabled to Matt
	if face_analyze:
		msg = osc_message_builder.OscMessageBuilder(address="/isMomentsEnabled")
		msg.add_arg(0)
		msg = msg.build()
		client.send(msg)

	ret, img = cam.read()
	# flip camera 90 degrees
	rotate = imutils.rotate_bound(img, 90)
	flipped = cv2.flip(rotate, 1)
	# convert image to grayscale
	gray_img = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
	# convert from bgr to rgb
	rgb_img = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
	# gray_img is the input grayscale image
	# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
	# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
	# minSize (optional) is the minimum rectangle size to be considered a face
	faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=6)
	# if no faces are detected
	# when faces are detected, the faces variable is an array of tuple, when no faces are detected the faces variable is an empty tuple
	if isinstance(faces, tuple):
		noface_counter+=1
		# reset 
		found_face = False
		face_analyze = False
		face_counter = 0
		avg_counter = 0
		emotion_text.clear()
		gender_text.clear()
		wearing_glasses.clear()
		shirt_color.clear()

	# camera found one or more faces
	else:
		# reset noFaces timer
		noface_counter = 0
		# focusing on a single face
		if found_face:
			x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
			# crop image so we only focus on this face
			cropped_img = gray_img[y1:y2, x1:x2]
			faces = face_detector.detectMultiScale(cropped_img, scaleFactor=1.3, minNeighbors=6)
			# is the face gone?
			if isinstance(faces, tuple):
				# print("FACE GONE")
				# tell matt we lost the face after a significant amount of tracking
				# if face_analyze:
				# 	msg = osc_message_builder.OscMessageBuilder(address="/lostFace")
				# 	msg = msg.build()
				# 	client.send(msg)
				found_face = False
				face_counter = 0
				face_analyze = False
				emotion_text.clear()
				shirt_color.clear()
				gender_text.clear()
				wearing_glasses.clear()
				# print("lost your face!")
			# face is still there
			else:
				# send coordinates to max so avatar looks at face
				# sendCoords(x,y,w,h)
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
					if face_counter > 5:
						# for when it's a solo person
						if len(faces) == 1:
							if random.random() < .33:
								face_analyze = True
							else:
								face_counter = 0
						else:
							face_analyze = True
				# we're ready to analyze this face!
				if face_analyze:
					# print("analyzing face!")
					x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), emotion_offsets)
					gray_face_og = gray_img[y1:y2, x1:x2]
					x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), gender_offsets)
					rgb_face_og = rgb_img[y1:y2, x1:x2]
					try:
						gray_face = cv2.resize(gray_face_og, (emotion_target_size))
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
					rect = dlib.rectangle(face_x, face_y, face_x+face_w, face_y+face_h)
					landmarks = predictor(gray_img, rect)
					landmarks = landmarks_to_np(landmarks)
					LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(flipped, landmarks)
					aligned_face = get_aligned_face(gray_img, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
					wearing_glasses.append(judge_eyeglass(aligned_face))

					avg_counter += 1

					# shirt color
					s_y = face_y + int(face_h * 1.65)
					s_h = face_h + int(face_h * 1.25)
					s_y2 = s_y+s_h
					# ensure shirt region doesn't extend outside of image
					rgb_height, rgb_width = rgb_img.shape[:2]
					if x2 >= rgb_width:
						x2 = rgb_width -1
					if s_y2 >= rgb_height:
						s_y2 = rgb_height -1
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
						avg_s += .2
						avg_v += avg_v * .12

						new_r, new_g, new_b = hsv_to_rgb(avg_h, avg_s, avg_v)
						print("new averages", new_r, new_g, new_b)
						# cv2.rectangle(flipped,(face_x,s_y), (face_x+face_w, s_y+face_h), (0,255,0), 2)
						# actual_name, closest_name = get_color_name( (int(new_r),int(new_g),int(new_b)) )
						color_name = ColorNames.findNearestImageMagickColorName((int(new_r),int(new_g),int(new_b)))
						# magik_name = ColorNames.findNearestImageMagickColorName((int(new_r),int(new_g),int(new_b)))
						# print(color_name)
						shirt_color.append(color_name)

					if avg_counter > 30:
						# if looking_for[looking_for_count] == "glasses" and max(set(wearing_glasses), key=wearing_glasses.count):
						# 	test_pass = True
						# elif looking_for[looking_for_count] == "man" and max(set(gender_text), key=gender_text.count) == "man":
						# 	test_pass = True
						# elif looking_for[looking_for_count] == "woman" and max(set(gender_text), key=gender_text.count) == "woman":
						# 	test_pass = True
						# elif looking_for[looking_for_count] == "light_shirt":
						# 	avg_shirt_color = max(set(shirt_color), key=shirt_color.count)
						# 	color_hex = ColorNames.ImageMagickColorMap[avg_shirt_color]
						# 	color_hex_num = color_hex.replace("#", "0x")
						# 	color_hex_num = int(color_hex_num, 16)
						# 	if color_hex_num > 0x708090:
						# 		test_pass = True
						# elif looking_for[looking_for_count] == "dark_shirt":
						# 	avg_shirt_color = max(set(shirt_color), key=shirt_color.count)
						# 	color_hex = ColorNames.ImageMagickColorMap[avg_shirt_color]
						# 	color_hex_num = color_hex.replace("#", "0x")
						# 	color_hex_num = int(color_hex_num, 16)
						# 	if color_hex_num < 0x708090:
						# 		test_pass = True
						# if test_pass:
						msg = osc_message_builder.OscMessageBuilder(address="/takeAPic")
						msg.add_arg(0)
						msg = msg.build()
						client.send(msg)
						# get timestamp
						ts = time.gmtime()
						timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
						fileName = "../faces/face" + timestamp + ".png"
						# convert image to 1:1 aspect ratio						
						x1 -= (x2-x1) * .25
						x2 += (x2-x1) * .25
						y1 -= (y2-y1) * .1
						y2 += (y2-y1) * .45
						flipped_h, flipped_w = flipped.shape[:2]
						if x1 < 0:
							x1 = 0
						if x2 >= flipped_w:
							x2 = flipped_w -1
						if y1 < 0:
							y1 = 0
						if y2 >= flipped_h:
							y2 = flipped_h -1
						portrait_img = flipped[int(y1):int(y2), int(x1):int(x2)]
						scale_percent = 225 # percent of original size
						bigger_width = int(portrait_img.shape[1] * scale_percent / 100)
						# bigger_height = int(bigger_width * 1.2)
						bigger_height = int(portrait_img.shape[0] * scale_percent / 100)
						dim = (bigger_width, bigger_height)
						# resize image
						resized = cv2.resize(portrait_img, dim, interpolation = cv2.INTER_CUBIC)
						aspect_ratio_h, aspect_ratio_w = resized.shape[:2]
						# crop image to be square
						new_height = aspect_ratio_w 
						final_img = resized[0:int(new_height), 0:aspect_ratio_w]
						# white padding around center
						final_final_img = cv2.copyMakeBorder(final_img, int(aspect_ratio_w*.05), int(aspect_ratio_w*.31), int(aspect_ratio_w*.18), int(aspect_ratio_w*.18), borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
						
						avg_shirt_color = max(set(shirt_color), key=shirt_color.count)
						shirt_list = re.findall('[A-Z][^A-Z]*', avg_shirt_color)
						if shirt_list != []:
							shirt_list = " ".join(shirt_list)
						else:
							shirt_list = avg_shirt_color
						shirt_list = ''.join([i for i in shirt_list if not i.isdigit()])
						# sad, surprise, happy, angry, neutral, disgust, and fear
						emotion_caption = max(set(emotion_text), key=emotion_text.count)
						if emotion_caption == "sad":
							new_emotion = sad_list[sad_counter]
							if sad_counter == len(sad_list)-1:
								sad_counter = 0
							else:
								sad_counter+=1
						elif emotion_caption == "happy":
							new_emotion = happy_list[happy_counter]
							if happy_counter == len(happy_list)-1:
								happy_counter = 0
							else:
								happy_counter+=1
						elif emotion_caption == "neutral":
							new_emotion = neutral_list[neutral_counter]
							if neutral_counter == len(neutral_list)-1:
								neutral_counter = 0
							else:
								neutral_counter+=1
						elif emotion_caption == "angry":
							new_emotion = angry_list[angry_counter]
							if angry_counter == len(angry_list)-1:
								angry_counter = 0
							else:
								angry_counter+=1
						elif emotion_caption == "fear":
							new_emotion = fear_list[fear_counter]
							if fear_counter == len(fear_list)-1:
								fear_counter = 0
							else:
								fear_counter+=1
						elif emotion_caption == "disgust":
							new_emotion = disgust_list[disgust_counter]
							if disgust_counter == len(disgust_list)-1:
								disgust_counter = 0
							else:
								disgust_counter+=1
						else:
							new_emotion = surprise_list[surprise_counter]
							if surprise_counter == len(surprise_list)-1:
								surprise_counter = 0
							else:
								surprise_counter+=1
						emotion_caption = emotion_list[emotion_list_counter].replace("(Emotion)", new_emotion)
						if emotion_list_counter == len(emotion_list)-1:
								emotion_list_counter = 0
						else:
							emotion_list_counter += 1
						second_caption = "Human in " + shirt_list + " garment"
						if max(set(wearing_glasses), key=wearing_glasses.count):
							emotion_caption += " and bespeckled eyes"
						pil_img = cv2.cvtColor(final_final_img,cv2.COLOR_BGR2RGB)
						pilimg = Image.fromarray(pil_img)
						draw = ImageDraw.Draw(pilimg)
						draw.text((aspect_ratio_w * .18, aspect_ratio_w + (aspect_ratio_w * .06)), emotion_caption, (0,0,0), font=ft_bold)
						draw.text((aspect_ratio_w * .18, aspect_ratio_w + (aspect_ratio_w * .12)), second_caption, (0,0,0), font=ft_color)
						draw.text((aspect_ratio_w * .18, aspect_ratio_w + (aspect_ratio_w * .2)), "2019", (0,0,0), font=ft_color)
						draw.text((aspect_ratio_w * .18, aspect_ratio_w + (aspect_ratio_w * .28)), "Collection of the artist", (0,0,0), font=ft_collection)
						cv2img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
						cv2.imwrite(fileName, cv2img)
						subprocess.call([r'C:/Users/student/Documents/GabeSanFran/take-my-pic/insta.bat', fileName, emotion_caption])
						test_pass = False
						num_of_pics += 1
						t_end = time.time() + 60 * 2
						tell_matt = time.time() + ((60 * 2) * .8)
						# while time.time() < t_end:
						# 	msg = osc_message_builder.OscMessageBuilder(address="/isMomentsEnabled")
						# 	arg = 1
						# 	if time.time() > tell_matt:
						# 		arg = 0
						# 	msg.add_arg(arg)
						# 	msg = msg.build()
						# 	client.send(msg)

						if num_of_pics >= 1:
							looking_for_count += 1
							if looking_for_count  >= len(looking_for):
								looking_for_count = 0
							num_of_pics = 0
						# exit the loop
						# ret = False
						# Reset everything
						avg_counter = 0
						emotion_text.clear()
						wearing_glasses.clear()
						shirt_color.clear()
						gender_text.clear()
						found_face = False
						face_analyze = False
						face_counter = 0
						# find_face_mode = False
				
		# still looking for a face to focus on
		else:	
			# if more than one face, loop through looking at face
			# if len(faces) > 1 and face_looking is False and find_face_mode is False:
			# 	face_looking = True
			# 	print("LOOKING MODE")
			np.random.shuffle(faces)
			for (x,y,w,h) in faces: 
				# if face_looking:
					# # look at each frame for three seconds
					# t_end = time.time() + 3
					# msg = osc_message_builder.OscMessageBuilder(address="/newFace")
					# msg = msg.build()
					# client.send(msg)
					# while time.time() < t_end:
						# sendCoords(x,y,w,h)
				# we are now focusing on one face
				# else:
				found_face = True;
				face_x, face_y, face_w, face_h = x,y,w,h
				# x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
				# crop image so we only focus on this face
				# cropped_img = gray_img[y1:y2, x1:x2]
				break

	cv2.imshow('test window', flipped)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break
 
# # end of program
cam.release()
cv2.destroyAllWindows()
