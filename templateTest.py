import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('res.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('templates/squiggle.jpg',0)
template = cv2.threshold(template, 60, 255, cv2.THRESH_BINARY_INV)[1]
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF)
threshold = 0.6
loc = np.where( res >= threshold)
numMatches = 0
for pt in zip(*loc[::-1]):
  numMatches += 1
  cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
 
print "found", numMatches, "matches"
cv2.namedWindow('MODDED', cv2.WINDOW_NORMAL)
cv2.imshow('MODDED', img_rgb)
#cv2.namedWindow('template', cv2.WINDOW_NORMAL)
#cv2.imshow('template', template)


k = cv2.waitKey(0)
cv2.destroyAllWindows()

