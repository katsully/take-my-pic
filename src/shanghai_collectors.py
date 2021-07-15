import cv2
import numpy as np
from pythonosc import osc_message_builder
from pythonosc import udp_client
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio
from imutils import rotate_bound
from utils.inference import apply_offsets
from random import randint
from time import time
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3,1920) # width
cam.set(4,1080)  # height
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

x1 = x2 = y1 = y2 = 0
flipped = {}
tracking_faces = False
selfie = False

# build udp_client for osc protocol
osc_client = udp_client.UDPClient("127.0.0.1", 8001)

def take_photo(address, *args):
    global x1, x2, y1, y2   # coordinates surrounding the face
    global flipped          # matrix with the camera view rotated and flipped
    global moment_time      # boolean stating whether avatar can do a 'moment'
    global selfie

    img_h, img_w = flipped.shape[:2]

    if selfie:
        remaining_y_space = (img_w - (y2 - y1))
        if y1 - (remaining_y_space/2) < 0:
            remaining_y_space -= y1
            y1 = 0
            y2 += remaining_y_space
        elif y2 + (remaining_y_space/2) > img_h:
            remaining_y_space -= (img_h - y2)
            y2 = img_h
            y1 -= remaining_y_space
        else:
            y2 += remaining_y_space / 2
            y1 -= remaining_y_space / 2
        crop_img = flipped[int(y1):int(y2), 0:int(img_w)]

        pil_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(pil_img)

        overlay = Image.open('CarliSelfie.png')

        pilimg.paste(overlay, (0,0), mask = overlay)

        final_img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)                     
        selfie = False

    else:
        # convert image to 1:1 aspect ratio                     
        x1 -= (x2-x1) * .3
        x2 += (x2-x1) * .3
        y1 -= (y2-y1) * .15
        y2 += (y2-y1) * .45
        if x1 < 0:
            x1 = 0
        if x2 >= img_w:
            x2 = img_w -1
        if y1 < 0:
            y1 = 0
        if y2 >= img_h:
            y2 = img_h -1
        portrait_img = flipped[int(y1):int(y2), int(x1):int(x2)]
        scale_percent = 225 # percent of original size
        bigger_width = int(portrait_img.shape[1] * scale_percent / 100)
        bigger_height = int(portrait_img.shape[0] * scale_percent / 100)
        dim = (bigger_width, bigger_height)
        # resize image
        resized = cv2.resize(portrait_img, dim, interpolation = cv2.INTER_CUBIC)
        aspect_ratio_h, aspect_ratio_w = resized.shape[:2]
        # crop image to be square
        final_img = resized[0:aspect_ratio_w, 0:aspect_ratio_w]     

    cv2.imwrite("../faces/photo.png", final_img) 
    print("saved photo")


def moment_done(address, *args):
    global tracking_faces

    tracking_faces = True

dispatcher = Dispatcher()
dispatcher.map("/photoAnimation", take_photo)
dispatcher.map("/momentDone", moment_done)

def moments_enabled(arg):
    msg = osc_message_builder.OscMessageBuilder(address="/isMomentsEnabled")
    msg.add_arg(arg)
    msg = msg.build()
    osc_client.send(msg)

async def face_finding():
    global cam
    global x1, x2, y1, y2
    global flipped
    global tracking_faces
    global selfie

    face_detector = cv2.CascadeClassifier('../trained_models/detection_models/haarcascade_frontalface_default.xml')

    # hyper-parameters for bounding boxes shape
    crop_offsets = (50, 70)
   
    capture_counter = 0     # keep track of capturing every third face
    selfie_counter = 1      # keep track of when we do selfie vs 'normal' pic
    tracking_faces = True   # whether we are tracking faces or letting the avatars do an animation (ie a 'moment')

    # counter for collecting the avg info about a person
    face_counter = 0
    found_face = False
    face_x = face_y = face_w = face_h = 0
    photo_ready = False

    # captions
    temp_file = open("captions.txt", "r", encoding="utf8")
    captions = [line.rstrip() for line in temp_file.readlines()]
    temp_file.close()
    caption_counter = 0

    if cam.isOpened(): # try to get the first frame
        ret, img = cam.read()   
    else:
        ret = False

    while(ret):
        ret, img = cam.read()

        # release program control back to the event loop
        await asyncio.sleep(0)
            
        # flip camera 90 degrees
        rotate = rotate_bound(img, 90)
        flipped = cv2.flip(rotate, 1)

        if tracking_faces:
            # tell Matt not to start animation
            moments_enabled(0)

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
            if len(faces) == 0:
                # reset 
                found_face = False
                photo_ready = False
                face_counter = 0

            # camera found one or more faces
            else:
                # focusing on a single face
                if found_face:
                    x1,x2,y1,y2 = apply_offsets((face_x, face_y, face_w, face_h), crop_offsets)
                    # crop image so we only focus on this face
                    cropped_img = gray_img[y1:y2, x1:x2]
                    faces = face_detector.detectMultiScale(cropped_img, scaleFactor=1.3, minNeighbors=6)
                    # is the face gone?
                    if isinstance(faces, tuple):
                        found_face = False
                        face_counter = 0
                        photo_ready = False
                    # face is still there
                    else:
                        # if we're still determining this face isn't someone quickly entering and exiting
                        if not photo_ready:
                            face_counter += 1
                            # face isn't just coming and going
                            if face_counter > 5:
                                photo_ready = True
                        # we're ready to analyze this face!
                        if photo_ready:
                            if capture_counter % 3 == 0:  
                                # tell matt to take a photo
                                if selfie_counter % 5 == 0:
                                    selfie = True
                                    print("sending to matt to take selfie pic")
                                    msg = osc_message_builder.OscMessageBuilder(address="/takeAPicSelfie")
                                else:
                                    print("sending to matt to take pic")
                                    msg = osc_message_builder.OscMessageBuilder(address="/takeAPic")

                                msg.add_arg(0)
                                msg = msg.build()
                                osc_client.send(msg)
                                
                                selfie_counter += 1

                                # stop searching for faces until matt takes photo
                                tracking_faces = False

                                caption = captions[caption_counter]
                                caption_counter += 1
                                if caption_counter >= len(captions):
                                    caption_counter = 0

                                title = caption.split("#")[0]
                                hashtag = caption.split("#")[1]

                                # send matt the title & hashtag
                                msg = osc_message_builder.OscMessageBuilder(address="/title")
                                msg.add_arg(title)
                                msg = msg.build()
                                osc_client.send(msg)    

                                msg = osc_message_builder.OscMessageBuilder(address="/hashtag")
                                msg.add_arg("#" + hashtag)
                                msg = msg.build()
                                osc_client.send(msg)    

                            capture_counter += 1

                            # Reset everything
                            found_face = False
                            photo_ready = False
                            face_counter = 0
                        
                # still looking for a face to focus on
                else:   
                    np.random.shuffle(faces)
                    for (x,y,w,h) in faces: 
                        found_face = True;
                        face_x, face_y, face_w, face_h = x,y,w,h
                        break
            # cv2.imshow("test window", flipped)
            k = cv2.waitKey(30 & 0xff)
            if k == 27:
                break
        # time right after photo is taken, where the avatar will do an animation/moment
        # elif moment_time:
        #     t_end = time() + 60
        #     while time() < t_end:
        #         moments_enabled(1)
        #     moments_enabled(0)
        #     tracking_faces = True
        #     moment_time = False

    
async def init_main():
    server = AsyncIOOSCUDPServer(
        ("127.0.0.1", 8002), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await face_finding()    # entering main loop of program

    transport.close()   # clean up serve endpoint

loop = asyncio.get_event_loop()
loop.run_until_complete(init_main())
    
# end of program
cam.release()
cv2.destroyAllWindows()