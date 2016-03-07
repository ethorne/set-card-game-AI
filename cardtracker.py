# Standard imports
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
    m = mse(imageA, imageB)  # MSE Increase  = less similar
    if m < 1000 :    return False
    else        :    return True

while True:
    _, frame    = cap.read();
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 85
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([180,sensitivity,255])
    frame   = imutils.resize(frame)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.erode  (mask, None, iterations=2)
    mask = cv2.dilate (mask, None, iterations=2)
    #cv2.imshow("Mask", mask)

    #Detect Contours
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index = 12
    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)

        if len(approx)==4 and perimeter > 400 and perimeter < 800 and w > 120 and h > 180:
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'Card ' + str(index) ,(x+15,y-10),1,1,(0,255,0))
                #cv2.putText(frame,str(h),(x+50,y-10),1,1,(0,255,0))
                roi=frame[y:y+h,x:x+w]
                
                if index == 12 :
                    cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
                    index = index - 1
                else:
                    for i in range(1, index):

                            # Compare current to previously saved images

                        img = cv2.imread('images/extracted/'+str(index+1)+'.png')
                        #cv2.imshow("saved" , img)
                        
                        height, width = img.shape[:2]
                        roi = cv2.resize(roi,(width, height), interpolation = cv2.INTER_CUBIC)
                        similar = compare_images(roi,img)
                        if similar == True:
                            cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
                            index = index - 1
                            
                #cv2.drawContours(frame,[cnt],-1,(0,0,255),2)
    #cv2.drawContours(frame, contours, -1, (255,0,0), 3)

    cv2.imshow("Contours" , frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:  break;

cv2.destroyAllWindows()
cap.release()