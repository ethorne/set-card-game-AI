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
		self.Shape = self.GetShape(cardImage)
		self.Color = self.GetColor(cardImage)
		self.Shade = self.GetShape(cardImage)
		self.Count = self.GetCount(cardImage)

	def __repr__(self):
		shapes = ['oval', 'diamond', 'squiggle', 'n/a']
		colors = ['red', 'green', 'purple', 'n/a']
		shades = ['solid', 'striped', 'outlined', 'n/a']
		
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

	def GetShape(self, image):
		imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		imageBlurred = cv2.GaussianBlur(imageGray, (5,5), 0)
		imageThresh = cv2.threshold(imageBlurred, 60, 255, cv2.THRESH_BINARY)[1]

		contours = cv2.findContours(imageThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0]

		for contour in contours:
			cv2.drawContours(image, [contour], -1, [0,0,0], 2)

		return None

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
			print('color ', color)

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

	def GetShade(self, image):
		# NOT IMPLEMENTED
		return None

	def GetCount(self, image):
		# NOT IMPLEMENTED
		return None