import cv2
import numpy as np

cap  = cv2.VideoCapture(0) # Import image, change to 1 for external usb cam


while (cap.isOpened()):
	ret, img = cap.read() # BGR Image feed from camera
	cv2.imshow('output',img)


	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # BGR to Gra
	#cv2.imshow('grayscale',img2)
	
	
	k=cv2.waitKey(10)   # Get input Key value
	if k == 27:		
		break;	# If key == 27 (Esc) exit
		

cap.release()
cv2.destroyAllWindows 