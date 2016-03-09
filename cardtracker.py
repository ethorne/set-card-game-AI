import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1);
 
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB):
    mseError = mse(imageA, imageB)      # Mean Squared Error Increase  = Less similar
    if mseError < 1000 : return True
    else               : return False

while True:
        
    # Get Image
    _, frame    = cap.read();
    index = 12          # Total Number of Cards we are able to detect
    
    # Transform Image to get desired values
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find Values in a certain range
    sensitivity = 85
    lowerwhite  = np.array([0,0,255-sensitivity])
    upperwhite  = np.array([180,sensitivity,255])

    # Increase extracted image quality, reduce blur
    frame   = imutils.resize(frame)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    mask    = cv2.inRange(hsv, lowerwhite, upperwhite)
    mask    = cv2.erode  (mask, None, iterations=2)
    mask    = cv2.dilate (mask, None, iterations=2)
    #cv2.imshow("Mask", mask)

    #Find Contours
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1,(0,0,255),2)
    
    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)

        # Make sure the image is of a card, by testing :
        # Height-Width ratio
        # Number of contours in bounding box
        # Total Perimeter of contours

        if len(approx)==4 and perimeter > 400 and perimeter < 1400 and w > 120 and h > 180:

                # Draw rectangle + text around found card
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'Card '       + str(index)                   ,(x+15,y-60),1,1,(0,255,0))
                #cv2.putText(frame,'Height : '   + str(h) + ', Width : '+ str(w),(x+15,y-40),1,1,(0,255,0))
                #cv2.putText(frame,'Perimeter : '+ str(perimeter)               ,(x+15,y-20),1,1,(0,255,0))
                
                # Specify that the stuff inside the contour is what we are interested in
                roi=frame[y:y+h,x:x+w]
                
                if index == 12 : # If this is the first card found, save it 
                    cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
                    index = index - 1

                else:
                    for i in range(1, index):

                        # Compare current to previously saved cards
                        img = cv2.imread('images/extracted/'+str(index+1)+'.png')

                        # Make sure the height and width match for MSE Calculation
                        height, width = img.shape[:2]
                        roi = cv2.resize(roi,(width, height), interpolation = cv2.INTER_CUBIC)
                        similar = compare_images(roi,img)

                        if similar == False:  
                            # If we think its a new card, save it
                            cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
                            index = index - 1
                        
    cv2.imshow("Contours" , frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:  break;

cv2.destroyAllWindows()
cap.release()