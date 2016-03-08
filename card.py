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
		self.Color = self.GetColor(cardImage)

		mod = self._preprocess(cardImage)

		# Count must be called before Shade
		#	because Count populates self._singleShape
		self.Count = self.GetCount(mod)

		self.Shade = self.GetShade(cardImage)
		self.Shape = self.GetShape(mod)

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

	def GetColor(self, image):
		# there should only be three colors
		#	white (card background)
		#	black (surface background)
		#	card color
		numColorsOnCard = 3

		# these values are determiend experimentally
		blackLimit = 50			# if the RGB vals are below this, color is black
		whiteLimit = 200		# if the RGB vals are above this, color is white
		purpleDifference = 35	# if abs(red-blue) is less than this, color is purple

		# convert image to RGB
		imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# TODO : saturate image before 

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

		# find dominant color that is not white or black
		for (percent, color) in zip(histogram, cluster.cluster_centers_):			
			red = color[0]
			green = color[1]
			blue = color[2]

			if (red < blackLimit and green < blackLimit and blue < blackLimit):
				# this is the black surface background
				continue

			if (red > whiteLimit and green > whiteLimit and blue > whiteLimit):
				# this is the white card background
				continue

			if (green > red and green > blue):
			 	return COLOR.Green

			if (abs(red - blue) < purpleDifference):
				return COLOR.Purple

			if (red > green and red > blue):
				return COLOR.Red

		return None

	def GetShape(self, image):
		return None
		shapeTemplates = ['tempaltes/oval.jpg',
						'tempaltes/squiggle.jpg',
						'tempaltes/diamond.jpg']

		shapeSize = (self._singleShape.shape[1], self._singleShape.shape[0])
		zeros = np.zeros(shapeSize)

		for templateLocation in shapeTemplates:
			print templateLocation
			template = cv2.imread(templateLocation, 0)
			resizedTemplate = cv2.resize(template, shapeSize, cv2.INTER_CUBIC)

			diff = zeros.copy()
			diff = cv2.absdiff(resizedTemplate, self._singleShape, diff)
			val = cv2.bitwise_and(diff, zeros)
			print 'val', val 
		return None

	def GetShade(self, image):
		
		return None

	def GetCount(self, image):
		count = 0
		contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0];		
		savedSingleShape = False
		for contour in contours:
			if cv2.contourArea(contour) < 500: # this number is subject to change based on what size the standard image will be
				continue
			
			x,y,w,h = cv2.boundingRect(contour)

			if not savedSingleShape:
				self._singleShape = image[y:(y+h), x:(x+w)] 
				savedSingleShape = True

			count += 1

		return count

	def _preprocess(self, image):
		# blur, contrast, denoise, and take inverse threshold
		mod = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		mod = cv2.GaussianBlur(mod, (5,5), 0)
		mod = self._increaseContrast(mod)

		mod = cv2.fastNlMeansDenoising(mod, None, 10, 21, 20)
		mod = cv2.threshold(mod, 60, 255, cv2.THRESH_BINARY_INV)[1]
				
		return mod

	def _increaseContrast(self, image):
		hist,bins = np.histogram(image.flatten(),256,[0,256])
		cdf = hist.cumsum()
		cdf_normalized = cdf * hist.max()/ cdf.max()
		
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')

		return cdf[image]




