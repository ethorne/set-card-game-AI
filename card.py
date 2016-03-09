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
		self._singleShapeMod = None
		self._singleShapeOrig = None
		self.isValid = False

		image = self._crop(cardImage, .08)

		# Count must be called first
		#	because Count populates self._singleShapeMod
		#	as well as self._singleShapeOrig
		self.Count = self.GetCount(image)
		if (self._singleShapeMod == None or self._singleShapeOrig == None):
			self.Shape = None
			self.Color = None
			self.Shade = None
			return

		self.Shape = self.GetShape()
		self.Color = self.GetColor() # must be called before shade
		self.Shade = self.GetShade()

		self.isValid = self.Count != None and self.Shape != None and self.Color != None and self.Shade != None

	def __repr__(self):
		return self.printCard()

	def printCard(self):
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

	# this code smells - seperate query and modifier
	# query - number of shapes on card (count)
	# modifier - populates necessary fields for shape, shade, and color
	#			 these fields are _singleShapeMod (a single countour line)
	#			 and _singleShapeOrig (a single shape from the card image, cropped)
	def GetCount(self, image):
		cny = cv2.Canny(image.copy(), 100, 250)

		count = 0
		savedSingleShape = False
		contours = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0];		
		for contour in contours:
			if cv2.contourArea(contour) < 2500: # this number is subject to change based on what size the standard image will be
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

		return count

	def GetColor(self):
		# there should only be two colors
		#	white (card background)
		#	card color
		numColorsOnCard = 2

		hueMax = 179.0 # max hue value for openCv
		# for an explanation of these fractions, refer to an HSV color wheel
		redRange = ( (3.0/4.0)*hueMax, (1.0/4.0)*hueMax )
		greenRange = ( (1.0/4.0)*hueMax, (1.0/2.0)*hueMax )
		purpleRange = ( (1.0/2.0)*hueMax, (3.0/4.0)*hueMax )
		
		# convert self._singleShapeOrig to RGB
		imageHsv = cv2.cvtColor(self._singleShapeOrig, cv2.COLOR_BGR2HSV)

		# reshape matrix into list of pixels
		imageHsv = imageHsv.reshape((imageHsv.shape[0] * imageHsv.shape[1], 3))
		
		# cluster with K-means clustering
		cluster = KMeans(n_clusters = numColorsOnCard)
		cluster.fit(imageHsv)

		# create histogram of num pixels for each color
		(histogram, notUsed) = np.histogram(cluster.labels_, bins = numColorsOnCard)

		# normalize histogram
		histogram = histogram.astype("float")
		histogram /= histogram.sum()

		# find color with highest saturation
		highestSat = -1.0e400
		nonWhiteColor = None
		self._nonWhitePercent = None
		for (percent, color) in zip(histogram, cluster.cluster_centers_):
			if (color[1] > highestSat):
				highestSat = color[1]
				nonWhiteColor = color
				self._nonWhitePercent = percent


		hue = nonWhiteColor[0]

		if (hue > redRange[0] or hue < redRange[1]):
			return COLOR.Red
		if (hue > greenRange[0] and hue < greenRange[1]):
			return COLOR.Green
		if (hue > purpleRange[0] and hue < purpleRange[1]):
			return COLOR.Purple

		return None

	def GetShade(self):
		# draw contours - if theres a lot, then its shaded
		# otherwise, refer to self._nonWhitePercent

		mod = self._increaseContrast(self._singleShapeOrig.copy())
		mod = self._crop(mod, .3)
		cny = cv2.Canny(mod, 0, 255)
		
		count = 0
		contours = cv2.findContours(cny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0];

		if (len(contours) > 10):
			return SHADE.Striped
		if self._nonWhitePercent > .5:
			return SHADE.Solid
		else:
			return SHADE.Outlined
		

		return None;

	def GetShape(self):
		# TODO : determine if we need to rotate the image
		#			by comparing ratios

		shapeTemplates = ['images/templates/oval.jpg',
						'images/templates/diamond.jpg',
						'images/templates/squiggle.jpg']

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

	def _increaseContrast(self, image):
		hist,bins = np.histogram(image.flatten(),256,[0,256])
		cdf = hist.cumsum()
		cdf_normalized = cdf * hist.max()/ cdf.max()
		
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')

		return cdf[image]

	def _crop(self, image, percent):
		widthDiff = image.shape[0] * percent
		heightDiff = image.shape[1] * percent
		return image[widthDiff:(image.shape[0] - widthDiff), heightDiff:(image.shape[1] - heightDiff)]

