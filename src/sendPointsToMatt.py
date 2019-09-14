from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
from imutils import face_utils


from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client


import dlib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# parameters for loading data and images
detection_model_path = '..\\trained_models\detection_models\haarcascade_frontalface_default.xml'
emotion_model_path = r"..\trained_models\emotion_models\fer2013_mini_XCEPTION.102-0.66.hdf5"
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# eyeglasses
predictor_path = "../data/shape_predictor_5_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

client = udp_client.UDPClient("127.0.0.1", 8001)

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
if video_capture.isOpened(): # try to get the first frame
    rval, bgr_image = video_capture.read()
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width)
    print(height)
else:
    rval = False
while rval:
    rval, bgr_image = video_capture.read()
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x, y, w, h = face_coordinates
        rect = dlib.rectangle(x,y,x+w,y+h)
        shape = predictor(gray_image, rect)
        shape = face_utils.shape_to_np(shape)

        top_of_nose = shape[3]
        # print(top_of_nose)
        cv2.circle(rgb_image, (top_of_nose[0], top_of_nose[1]), 20, 100)
        msg = osc_message_builder.OscMessageBuilder(address="/LookAt")
        xPos = top_of_nose[0]/width
        yPos = top_of_nose[1]/height
        msg.add_arg(xPos)
        msg.add_arg(yPos)
        msg = msg.build()
        client.send(msg)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
