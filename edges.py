import cv2
import argparse

screenRes = 1280, 800
argParser = argparse.ArgumentParser()

def configArgParser():
	argParser.add_argument('-i', '--image', required=True, help='path to input image')	


def outline(image, contour, color):
	cv2.drawContours(image, [contour], -1, color, 2)

def main():
	windowNames = ['original image', 'modified image']
	args = vars(argParser.parse_args())

	image = cv2.imread(args['image'])

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5,5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

	contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0]

	for contour in contours:
		outline(image, contour, (255,255,255))

	cv2.namedWindow(windowNames[0], cv2.WINDOW_NORMAL)
	cv2.imshow(windowNames[0], image);

	cv2.namedWindow(windowNames[1], cv2.WINDOW_NORMAL)
	cv2.imshow(windowNames[1], thresh);

	k = cv2.waitKey(0)
	cv2.destroyAllWindows()

configArgParser()
main()