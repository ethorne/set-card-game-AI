import cv2
import argparse
from card import Card

argParser = argparse.ArgumentParser()

def configArgParser():
	argParser.add_argument("-i", "--image", required=True, help="path to input image")

def main():
	windowNames = ["original image", "modified image"]
	args = vars(argParser.parse_args())

	setCard = Card(cv2.imread(args["image"]))
	print(setCard)

	# cv2.namedWindow(windowNames[0], cv2.WINDOW_NORMAL)
	# cv2.imshow(windowNames[0], image);

	# cv2.namedWindow(windowNames[1], cv2.WINDOW_NORMAL)
	# cv2.imshow(windowNames[1], thresh);

	# k = cv2.waitKey(0)
	# cv2.destroyAllWindows()

configArgParser()
main()