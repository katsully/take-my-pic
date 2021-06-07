import cv2
from keras.models import load_model
import numpy as np
import dlib
from pythonosc import osc_message_builder
from pythonosc import udp_client
import time
from imutils import face_utils
import imutils
import random
import re

from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from utils.kats_helper import landmarks_to_np
from utils.kats_helper import rgb_to_hsv
from utils.kats_helper import hsv_to_rgb
from utils.kats_helper import ColorNames

from PIL import Image
from PIL import ImageDraw

cam = cv2.VideoCapture(0)
# cv2.namedWindow("insta", flags=cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow("insta", 2160, 0)
# cv2.setWindowProperty("insta", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


cam.set(3,1920)	# width
cam.set(4,1080)  # height


face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
gender_labels = get_labels('imdb')

# counter for collecting the avg info about a person
avg_counter = 0
face_counter = 0
emotion_text = []
gender_text = []
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

tracking_faces = True

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

if cam.isOpened(): # try to get the first frame
	ret, img = cam.read()	
else:
    ret = False

while(ret):
	ret, img = cam.read()
	
	if tracking_faces:
		# flip camera 90 degrees
		# rotate = imutils.rotate_bound(img, 90)
		# flipped = cv2.flip(rotate, 1)
		# convert image to grayscale
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# convert from bgr to rgb
		rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# gray_img is the input grayscale image
		# scaleFactor (optional) is specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid
		# minNeighbors (optional) is specifying how many neighbors each candidate rectangle show have, to retain it. A higher number gives lower false positives
		# minSize (optional) is the minimum rectangle size to be considered a face
		faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=6)
		
		# if no faces are detected
		# when faces are detected, the faces variable is an array of tuple, when no faces are detected the faces variable is an empty tuple
		# if isinstance(faces, tuple):
		if len(faces) == 0:
			# reset 
			print("no faces found")
			found_face = False
			face_analyze = False
			face_counter = 0
			avg_counter = 0
			emotion_text.clear()
			gender_text.clear()
			# wearing_glasses.clear()
			shirt_color.clear()

		# camera found one or more faces
		else:
			# focusing on a single face
			# print("found_face is " + str(found_face))
			if found_face:
				print("found a face")
				x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
				# crop image so we only focus on this face
				cropped_img = gray_img[y1:y2, x1:x2]
				faces = face_detector.detectMultiScale(cropped_img, scaleFactor=1.3, minNeighbors=6)
				# is the face gone?
				if isinstance(faces, tuple):
					print("face gone")
					found_face = False
					face_counter = 0
					face_analyze = False
					emotion_text.clear()
					shirt_color.clear()
					gender_text.clear()
				# face is still there
				else:
					# if we're still determining this face isn't someone quickly entering and exiting
					if not face_analyze:
						face_counter += 1
						# face isn't just coming and going
						if face_counter > 5:
							face_analyze = True
					# we're ready to analyze this face!
					if face_analyze:
						print("analyzing face")
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

						avg_counter += 1

						# shirt color
						s_y = face_y + int(face_h * 1.65)
						s_h = face_h + int(face_h * 1.25)
						s_y2 = s_y+s_h
						# ensure shirt region doesn't extend outside of image
						rgb_height, rgb_width = rgb_img.shape[:2]
						if x2 >= rgb_width:
							x2 = rgb_width - 2
						if s_y2 >= rgb_height:
							s_y2 = rgb_height - 2
						if x1 < 0:
							x1 = 0
						shirt_region = rgb_img[s_y:int(s_y2), x1:x2]

						if type(shirt_region) is not 'NoneType':
							try:
								shirt_image = Image.fromarray(shirt_region, 'RGB')
							except: 
								print("shirt error")
								continue
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
							
							# up the saturation and brightness
							avg_s += avg_s * .2
							avg_v += avg_v * .12

							new_r, new_g, new_b = hsv_to_rgb(avg_h, avg_s, avg_v)
							# print("new averages", new_r, new_g, new_b)
							color_name = ColorNames.findNearestImageMagickColorName((int(new_r),int(new_g),int(new_b)))
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
							# msg = osc_message_builder.OscMessageBuilder(address="/takeAPic")
							# msg.add_arg(0)
							# msg = msg.build()
							# osc_client.send(msg)
							# get timestamp
							ts = time.gmtime()
							timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
							fileName = "../faces/face" + timestamp + ".png"
							print(fileName)
							# convert image to 1:1 aspect ratio						
							x1 -= (x2-x1) * .25
							x2 += (x2-x1) * .25
							y1 -= (y2-y1) * .1
							y2 += (y2-y1) * .45
							flipped_h, flipped_w = img.shape[:2]
							if x1 < 0:
								x1 = 0
							if x2 >= flipped_w:
								x2 = flipped_w -1
							if y1 < 0:
								y1 = 0
							if y2 >= flipped_h:
								y2 = flipped_h -1
							portrait_img = img[int(y1):int(y2), int(x1):int(x2)]
							scale_percent = 225 # percent of original size
							bigger_width = int(portrait_img.shape[1] * scale_percent / 100)
							bigger_height = int(portrait_img.shape[0] * scale_percent / 100)
							dim = (bigger_width, bigger_height)
							# resize image
							resized = cv2.resize(portrait_img, dim, interpolation = cv2.INTER_CUBIC)
							aspect_ratio_h, aspect_ratio_w = resized.shape[:2]
							# crop image to be square
							new_height = aspect_ratio_w 
							final_img = resized[0:int(new_height), 0:aspect_ratio_w]
							uniform_size = cv2.resize(final_img, (600,600))
							new_height, new_width = uniform_size.shape[:2]
							# white padding around center
							final_final_img = cv2.copyMakeBorder(uniform_size, int(new_width*.05), int(new_width*.31), int(new_width*.18), int(new_width*.18), borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
							
							avg_shirt_color = max(set(shirt_color), key=shirt_color.count)
							shirt_list = re.findall('[A-Z][^A-Z]*', avg_shirt_color)
							if shirt_list != []:
								shirt_list = " ".join(shirt_list)
							else:
								shirt_list = avg_shirt_color
							shirt_list = ''.join([i for i in shirt_list if not i.isdigit()])
							# sad, surprise, happy, angry, neutral, disgust, and fear
							# emotion_caption = max(set(emotion_text), key=emotion_text.count)
							# if emotion_caption == "sad":
							# 	new_emotion = sad_list[sad_counter]
							# 	if sad_counter == len(sad_list)-1:
							# 		sad_counter = 0
							# 	else:
							# 		sad_counter+=1
							# elif emotion_caption == "happy":
							# 	new_emotion = happy_list[happy_counter]
							# 	if happy_counter == len(happy_list)-1:
							# 		happy_counter = 0
							# 	else:
							# 		happy_counter+=1
							# elif emotion_caption == "neutral":
							# 	new_emotion = neutral_list[neutral_counter]
							# 	if neutral_counter == len(neutral_list)-1:
							# 		neutral_counter = 0
							# 	else:
							# 		neutral_counter+=1
							# elif emotion_caption == "angry":
							# 	new_emotion = angry_list[angry_counter]
							# 	if angry_counter == len(angry_list)-1:
							# 		angry_counter = 0
							# 	else:
							# 		angry_counter+=1
							# elif emotion_caption == "fear":
							# 	new_emotion = fear_list[fear_counter]
							# 	if fear_counter == len(fear_list)-1:
							# 		fear_counter = 0
							# 	else:
							# 		fear_counter+=1
							# elif emotion_caption == "disgust":
							# 	new_emotion = disgust_list[disgust_counter]
							# 	if disgust_counter == len(disgust_list)-1:
							# 		disgust_counter = 0
							# 	else:
							# 		disgust_counter+=1
							# else:
							# 	new_emotion = surprise_list[surprise_counter]
							# 	if surprise_counter == len(surprise_list)-1:
							# 		surprise_counter = 0
							# 	else:
							# 		surprise_counter+=1
							# emotion_caption = emotion_list[emotion_list_counter].replace("(Emotion)", new_emotion)
							# if emotion_list_counter == len(emotion_list)-1:
							# 		emotion_list_counter = 0
							# else:
							# 	emotion_list_counter += 1
							# second_caption = second_caption_titles[second_caption_counter] + shirt_list.capitalize() 
							# if second_caption_counter == len(second_caption_titles)-1:
							# 		second_caption_counter = 0
							# else:
							# 	second_caption_counter += 1

							pil_img = cv2.cvtColor(final_final_img,cv2.COLOR_BGR2RGB)
							pilimg = Image.fromarray(pil_img)
							# draw = ImageDraw.Draw(pilimg)
							# draw.text((new_width * .18, new_width + (new_width * .07)), emotion_caption, (0,0,0), font=ft_bold)
							# draw.text((new_width * .18, new_width + (new_width * .14)), second_caption, (0,0,0), font=ft_color)
							# draw.text((new_width * .18, new_width + (new_width * .21)), "2019", (0,0,0), font=ft_color)
							# draw.text((new_width * .18, new_width + (new_width * .28)), "Collection of the artist", (0,0,0), font=ft_collection)
							cv2img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
							# # save file to faces database
							cv2.imwrite(fileName, cv2img)
							

							# flash_done = False
							# gabe_flash_counter = 12
							# flash_pause = True
							# tracking_faces = False

							# Reset everything
							avg_counter = 0
							emotion_text.clear()
							shirt_color.clear()
							gender_text.clear()
							found_face = False
							face_analyze = False
							face_counter = 0
					
			# still looking for a face to focus on
			else:	
				print("in else statement")
				np.random.shuffle(faces)
				for (x,y,w,h) in faces: 
					found_face = True;
					cv2.rectangle(img,(x,y),(x+w,y+h),(200,200,0),2)
					print("found face is true")
					face_x, face_y, face_w, face_h = x,y,w,h
					break

	# flip insta grid 90 degrees
	# rotate_insta = imutils.rotate_bound(insta_grid,90)
	cv2.imshow("test window", img)
	# cv2.imshow("insta", rotate_insta)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break


# end of program
cam.release()
cv2.destroyAllWindows()