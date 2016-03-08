import cv2
import numpy as np
import argparse

def increaseContrast(img):
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()
	
	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')

	return cdf[img]

# this doesn't work right now... need to get some help here as to why not
def brighten(img):
	imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(imgHsv)
	v += 50
	print v
	imgHsv = cv2.merge((h, s, v))
	return cv2.cvtColor(imgHsv, cv2.COLOR_HSV2BGR)

def extractShape(img):
	img2 = cv2.GaussianBlur(img, (5,5), 0)
	img2 = increaseContrast(img)

	img2 = cv2.fastNlMeansDenoising(img2, None, 10, 21, 20)
	img2 = cv2.threshold(img2, 60, 255, cv2.THRESH_BINARY_INV)[1]
			
	return img2

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(argParser.parse_args()) 	

img = extractShape(cv2.imread(args["image"], 0))
colorImage = cv2.imread(args["image"])

contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
moddedImg = np.zeros(img.shape)
count = 0
for cnt in contours:
	if cv2.contourArea(cnt) < 1000: # this number is subject to change based on what size the standard image will be
		continue
	singlShape = np.zeros(img.shape)
	cv2.drawContours(colorImage, [cnt], -1, [255,0,0], 2)
	count += 1
print "found", count, "shapes"



cv2.namedWindow('MODDED', cv2.WINDOW_NORMAL)
cv2.imshow('MODDED', colorImage)

k = cv2.waitKey(0)
cv2.destroyAllWindows()

