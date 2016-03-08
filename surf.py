import cv2
import numpy as np
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(argParser.parse_args())

img = cv2.imread(args["image"])
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.SURF(gra400)

kp, des = surf.detectAndCompute(img,None)

img2 = cv2.drawKeypoints(img,kp,None,(0,255,0),4)

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.imshow('window', img2);

k = cv2.waitKey(0)
cv2.destroyAllWindows()