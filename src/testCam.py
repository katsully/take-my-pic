import cv2
import imutils

cam = cv2.VideoCapture(0)
cam.set(3,1920)	# width
cam.set(4,1080)  # height
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

if cam.isOpened(): # try to get the first frame
	ret, img = cam.read()
else:
    ret = False

while(ret):
	ret, img = cam.read()
	# flip camera 90 degrees
	rotate = imutils.rotate_bound(img, 90)
	flipped = cv2.flip(rotate, 1)

	cv2.imshow('test window', flipped)
	k = cv2.waitKey(30 & 0xff)
	if k == 27: 	# press ESC to quit
		break
 
# end of program
cam.release()
cv2.destroyAllWindows()
