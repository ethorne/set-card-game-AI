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

	
	for i in range(1,15):
		cardName = 'testImages/' + str(i) + '.png';
		cardImage = cv2.imread(cardName)
		if cardImage == None:
			print 'Could not read image at ' + cardName
			continue
		print cardName
		start = timeit.default_timer()
		setCard = Card(cardImage)
		stop = timeit.default_timer()
		print(setCard)
		print "\nTIME ELAPSED : ", (stop - start), " s"
		print "\n------------------------------\n"

#configArgParser()
main()