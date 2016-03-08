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
		# Count must be called first
		#	because Count populates self._singleShapeMod
		#	as well as self._singleShapeOrig
		self.Count = self.GetCount(cardImage)
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
		mod = self._preprocess(image)

		count = 0
		contours = cv2.findContours(mod.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0];		
		savedSingleShape = False

		for contour in contours:
			if cv2.contourArea(contour) < 500: # this number is subject to change based on what size the standard image will be
				continue
			
			x,y,w,h = cv2.boundingRect(contour)
			
			if not savedSingleShape:
				self._singleShapeMod = mod[y:(y+h), x:(x+w)]
				self._singleShapeOrig = image[y:(y+h), x:(x+w)]
				savedSingleShape = True

			count += 1

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

	def _mse(self, imageA, imageB):
		err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
		err /= float(imageA.shape[0] * imageA.shape[1])
		
		return err

