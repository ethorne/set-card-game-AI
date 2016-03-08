#from enum import enum				<--- consider using this for easy iteration over attributes
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2

class SHAPE():
	Oval = 0
	Diamond = 1
	Squiggle = 2

class COLOR():
	Red = 0
	Green = 1
	Purple = 2

class SHADE():
	Solid = 0
	Striped = 1
	Outlined = 2

class Card():
	""" Represents a card for the game Set """

	def __init__(self, cardImage):
		image = self._crop(cardImage, .08)

		# Count must be called first
		#	because Count populates self._singleShapeMod
		#	as well as self._singleShapeOrig
		self.Count = self.GetCount(image)
		self.Shape = self.GetShape()
		self.Color, self.Shade = self.GetColorAndShade()

	def __repr__(self):
		shapes = ['oval', 'diamond', 'squiggle']
		colors = ['red', 'green', 'purple']
		shades = ['solid', 'striped', 'outlined']
		
		ret = 'Shape:\t'
		if (self.Shape != None):
			ret += shapes[self.Shape]

		ret += '\nColor:\t'
		if (self.Color != None):
			ret += colors[self.Color]

		ret += '\nShade:\t'
		if (self.Shade != None):
			ret += shades[self.Shade]

		ret += '\nCount:\t' + str(self.Count)
		return ret

	def GetColorAndShade(self):
		# there should only be two colors
		#	white (card background)
		#	card color
		numColorsOnCard = 2

		# these values are determiend experimentally
		whiteLimit = 200		# if the RGB vals are above this, color is white
		purpleDifference = 35	# if abs(red-blue) is less than this, color is purple

		# convert self._singleShapeOrig to RGB
		imageRgb = cv2.cvtColor(self._singleShapeOrig, cv2.COLOR_BGR2RGB)

		# reshape matrix into list of pixels
		imageRgb = imageRgb.reshape((imageRgb.shape[0] * imageRgb.shape[1], 3))
		
		# cluster with K-means clustering
		cluster = KMeans(n_clusters = numColorsOnCard)
		cluster.fit(imageRgb)

		# create histogram of num pixels for each color
		(histogram, notUsed) = np.histogram(cluster.labels_, bins = numColorsOnCard)

		# normalize histogram
		histogram = histogram.astype("float")
		histogram /= histogram.sum()

		dominantColor = None
		shade = None
		# find dominant color that is not white
		for (percent, color) in zip(histogram, cluster.cluster_centers_):
			red = color[0]
			green = color[1]
			blue = color[2]

			if (red > whiteLimit and green > whiteLimit and blue > whiteLimit):
				# this is the white card background
				continue

			if (green > red and green > blue):
				dominantColor = COLOR.Green
				shade = percent
				break

			if (abs(red - blue) < purpleDifference):
				dominantColor = COLOR.Purple
				shade = percent
				break

			if (red > green and red > blue):
				dominantColor = COLOR.Red
				shade = percent
				break

		print ' --- dominant color percent --- ', shade	

		if (shade < 0.20):
			shade = SHADE.Outlined
		elif (shade < 0.50):
			shade = SHADE.Striped
		else:
			shade = SHADE.Solid

		return dominantColor, shade

	def GetCount(self, image):
		cny = cv2.Canny(image.copy(), 225, 250)

		count = 0
		savedSingleShape = False
		contours = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0];		
		for contour in contours:
			if cv2.contourArea(contour) < 500: # this number is subject to change based on what size the standard image will be
				continue

			if not savedSingleShape:
				box = cv2.cv.BoxPoints(cv2.minAreaRect(contour))
				box = np.int0(box)

				# draw a thick line around the contour we are interested in
				cv2.drawContours(cny, [contour], 0, [255,255,255], 2)

				self._singleShapeMod = self._fourPointTransform(cny, box)
				self._singleShapeOrig = self._fourPointTransform(image, box)
				savedSingleShape = True

			count += 1


		# cny = cv2.Canny(image.copy(), 225, 250)

		# cv2.namedWindow('1', cv2.WINDOW_NORMAL)
		# cv2.imshow('1', self._singleShapeMod)
		# cv2.namedWindow('2', cv2.WINDOW_NORMAL)
		# cv2.imshow('2', self._singleShapeOrig)
		# cv2.moveWindow('1', 0, 0)
		# cv2.moveWindow('2', 200, 0)
		# k = cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return count

	def GetShape(self):
		shapeTemplates = ['templates/oval.jpg',
						'templates/diamond.jpg',
						'templates/squiggle.jpg']

		shapeSize = (self._singleShapeMod.shape[1], self._singleShapeMod.shape[0])
		mse = 1.0e400
		shape = None;
		for i in range(len(shapeTemplates)):
			tmp = cv2.imread(shapeTemplates[i], 0)
			tmp = cv2.resize(tmp, shapeSize, interpolation=cv2.INTER_CUBIC)
			newMse = self._mse(tmp, self._singleShapeMod)
			if newMse < mse:
				shape = i
				mse = newMse
		return shape

	# UNUSED - remove?
	def _preprocess(self, image):
		# blur, contrast, denoise, and take inverse threshold
		mod = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		mod = cv2.GaussianBlur(mod, (5,5), 0)
		mod = self._increaseContrast(mod)

		mod = cv2.fastNlMeansDenoising(mod, None, 10, 21, 20)
		mod = cv2.threshold(mod, 60, 255, cv2.THRESH_BINARY_INV)[1]
				
		return mod

	# UNUSED - remove?
	def _increaseContrast(self, image):
		hist,bins = np.histogram(image.flatten(),256,[0,256])
		cdf = hist.cumsum()
		cdf_normalized = cdf * hist.max()/ cdf.max()
		
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')

		return cdf[image]

	def _mse(self, imageA, imageB):
		err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
		err /= float(imageA.shape[0] * imageA.shape[1])
		
		return err

	# orders points (tl, tr, br, bl) for _fourPointFreeTransform
	def _orderPoints(self, pts):
		rect = np.zeros((4, 2), dtype = "float32")
	
		# top-left (tl) point has min sum
		# bottom-right (br point has max sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]
	 
		# top-right (tr) point has min difference
		# bottom-left (br) has max difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]
	 
	 	return rect

	def _fourPointTransform(self, image, pts):
		rect = self._orderPoints(pts)
		(tl, tr, br, bl) = rect
	 
		# compute the width of the new image
		# 	max( distance between br and bl, distance between tr and tl)

		widthTop = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		widthBottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		width = max(int(widthTop), int(widthBottom))
	 
	 	# same deal for height
		heightLeft = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		heightRight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		height = max(int(heightLeft), int(heightRight))
	 
		# now that we have the dimensions of the new image, construct
		# the set of destination points to obtain a "birds eye view",
		# (i.e. top-down view) of the image, again specifying points
		# in the top-left, top-right, bottom-right, and bottom-left
		# order
		dst = np.array([
			[0, 0],
			[width - 1, 0],
			[width - 1, height - 1],
			[0, height - 1]], dtype = "float32")
	 
		# compute the perspective transform matrix and then apply it
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (width, height))
	 
		# return the warped image
		return warped

	def _crop(self, image, percent):
		widthDiff = image.shape[0] * percent
		heightDiff = image.shape[1] * percent
		return image[widthDiff:(image.shape[0] - widthDiff), heightDiff:(image.shape[1] - heightDiff)]

