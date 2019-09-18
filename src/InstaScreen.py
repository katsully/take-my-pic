from bs4 import BeautifulSoup
import selenium.webdriver as webdriver
from PIL import Image
import requests
import cv2
import glob
import os
import numpy as np

url = 'http://instagram.com/lookingtogether/'
driver = webdriver.Chrome()
driver.get(url)

soup = BeautifulSoup(driver.page_source, features="html5lib")

all_photos = []

# adding photo from other machine to our database
def add_new_photo():
	newest_img = soup.find_all('img')[1]
	image_url = image['src']
	img = Image.open(requests.get(image_url, stream = True).raw)
	ts = time.gmtime()
	timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
	fileName = "../faces/face" + timestamp + ".png"
	cv2.imwrite(fileName, img)

# no longer need insta
driver.quit()

all_photos = glob.glob('../faces/*')
most_recent = max(all_photos, key=os.path.getctime)
print(most_recent)
recent_image = cv2.imread(most_recent)
recent_height, recent_width = recent_image.shape[:2]

# go through other photos
rows = []
for idx,filename in enumerate(all_photos[1:]):
	if idx % 5 == 0:
		if idx != 0:
			rows.append(smaller_pic)
		row_concat = cv2.imread(filename)
		smaller_pic = cv2.resize(row_concat, (225, 225), interpolation=cv2.INTER_AREA)
	else:
		temp_img = cv2.imread(filename)
		temp_smaller = cv2.resize(temp_img, (225, 225), interpolation=cv2.INTER_AREA)
		smaller_pic = np.concatenate((smaller_pic, temp_smaller), axis=1)

it = iter(rows)
for row1,row2 in zip(it, it):
	grid_images = np.concatenate((row1, row2), axis=0)


grid_height, grid_width = grid_images.shape[:2]
padding = int((grid_width - recent_width) / 2)
print("OG IMG", recent_width)
print("grid width", grid_width)
print("padding", padding)
# white padding around center of most recent image
padded_recent_img = cv2.copyMakeBorder(recent_image, 0, 0, padding, padding, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))

final_img = np.concatenate((padded_recent_img, grid_images), axis=0)


cv2.imshow("insta", final_img)



cv2.waitKey(0)
cv2.destroyAllWindows()

	