from PIL import Image
import cv2
import glob
import os
import time
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

insta_font = ImageFont.truetype(font="fonts/Mark Simonson Proxima Nova Light.otf",size=60)

def gabe_flash():
	return 255 * np.ones(shape=[1080, 1920, 3], dtype=np.uint8)


def update_screen():
	all_photos = glob.glob('../faces/*')
	all_photos.sort(key=os.path.getmtime)
	# most_recent = max(all_photos, key=os.path.getctime)
	recent_image = cv2.imread(all_photos[-1])
	h, w = recent_image.shape[:2]
	cropped = recent_image[0:h,int(w*.08):int(w*.92)]
	c_h, c_w = cropped.shape[:2]
	new_height = int((c_h*(1080/c_w)) - (1080*.1))
	resized_img = cv2.resize(cropped, (1080, new_height))
	recent_height, recent_width = resized_img.shape[:2]

	smaller_pic_size = int(1080/3)
	resized_pics = []
	for i in range(3):
		pic = cv2.imread(all_photos[-i-2])
		pic = cv2.copyMakeBorder(pic, 0, int(1080*.1), 0, 0, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
		resized_pics.append(cv2.resize(pic, (smaller_pic_size, smaller_pic_size)))
	
	grid_img = np.concatenate((resized_pics[0], resized_pics[1]), axis=1)
	grid_img = np.concatenate((grid_img, resized_pics[2]), axis=1)

	# white padding around center of most recent image
	grid_height, grid_width = grid_img.shape[:2]
	padding = int((grid_width - recent_width) / 2)
	vert_padding = int((1920 - (grid_height + recent_height)) / 2)
	if padding < (grid_width - recent_width) / 2:
		padded_recent_img = cv2.copyMakeBorder(resized_img, vert_padding, vert_padding, padding, padding+1, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
	else:
		padded_recent_img = cv2.copyMakeBorder(resized_img, vert_padding, vert_padding, padding, padding, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

	
	final_img = np.concatenate((padded_recent_img, grid_img), axis=0)

	# write instagram caption
	pil_img = cv2.cvtColor(final_img,cv2.COLOR_BGR2RGB)
	pilimg = Image.fromarray(pil_img)
	draw = ImageDraw.Draw(pilimg)
	insta_account = "thecollectorsgallery"
	text_length = insta_font.getsize(insta_account)[0]
	draw.text((1080/2 - text_length/2, 1920*.04), insta_account, (0,0,0), font=insta_font)
	cv2img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)

	final_img = cv2.line(cv2img, (0,int(1920*.78)), (1080,int(1920*.78)), (200,200,200), 2) 

	return final_img	


	