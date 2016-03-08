import cv2
import argparse
import timeit
from card import Card

argParser = argparse.ArgumentParser()

def configArgParser():
	argParser.add_argument("-i", "--image", required=True, help="path to input image")

def main():
	windowNames = ["original image", "modified image"]
	args = vars(argParser.parse_args())

	start = timeit.default_timer()
	setCard = Card(cv2.imread(args["image"]))
	stop = timeit.default_timer()
	print(setCard)

	print "\nTIME ELAPSED : ", (stop - start), " s"

configArgParser()
main()