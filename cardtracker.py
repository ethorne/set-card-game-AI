import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

 
class card:
    
    MIN_HEIGHT    = 120
    MIN_WIDTH     = 180
    MIN_PERIMETER = 400
    MAX_PERIMETER = 1400
    
    def __init__(self):
        self.x = 0;
        self.y = 0;
        self.w = 0;
        self.h = 0;
        self.perimeter = 0;
        self.approxEdges = 0;
        #self.cardImage = ();
        # Will contain card image as numpy array

    def updateDimensions(self, cnt):
        self.perimeter = cv2.arcLength(cnt,True)
        self.approxEdges = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        self.x,self.y,self.w,self.h = cv2.boundingRect(cnt)

    def isCard(self, contours):
    # Make sure the image is of a card, by testing :
        # Height-Width ratio
        # Number of contours in bounding box
        # Total Perimeter of contours

        for cnt in contours:
            self.updateDimensions(cnt)
            
            if          len(self.approxEdges) == 4:
                if      self.w > self.MIN_WIDTH             and self.h > self.MIN_HEIGHT:
                    if  self.perimeter > self.MIN_PERIMETER and self.perimeter < self.MAX_PERIMETER :
                        return True
            else:
                return False

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compareImages(imageA, imageB):
    mseError = mse(imageA, imageB)      # Mean Squared Error Increase  = Less similar
    if mseError < 1000 : return True
    else               : return False

def createMask(frame):
    # Transform Image to get desired values
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Find Values in a certain range
    sensitivity = 85
    lowerwhite  = np.array([0,0,255-sensitivity])
    upperwhite  = np.array([180,sensitivity,255])

    # Increase extracted image quality, reduce blur
    frame       = imutils.resize(frame)
    blurred     = cv2.GaussianBlur(frame, (11, 11), 0)
    mask        = cv2.inRange(hsv, lowerwhite, upperwhite)
    mask        = cv2.erode  (mask, None, iterations=2)
    mask        = cv2.dilate (mask, None, iterations=2)
    #cv2.imshow("Mask", mask)
    return mask

def findContours(mask):
    #Find Contours
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1,(0,0,255),2)
    #cv2.imshow("Contours" , frame)
    return contours

def drawBoundingBox(x,y,w,h, index,perimeter):
    
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(frame,'Card '       + str(index)                   ,(x+15,y-60),1,1,(0,255,0))
    #cv2.putText(frame,'Height : '   + str(h) + ', Width : '+ str(w),(x+15,y-40),1,1,(0,255,0))
    #cv2.putText(frame,'Perimeter : '+ str(perimeter)               ,(x+15,y-20),1,1,(0,255,0))
    
def makeSameSize(imageA,imageB):
    
    # NOTE : Resizes imageA size of imageB

    height, width = imageB.shape[:2]
    imageA = cv2.resize(imageA,(width, height), interpolation = cv2.INTER_CUBIC)
    return imageA

def saveImage(roi,index):

    # Should be getting a cardObj index and roi, and based on comparisions should update the card.cardImage np.array

    if index == 12 : # If this is the first card found, save it 
        cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
        index = index - 1

    else:
        for i in range(1, index):

            savedImage = cv2.imread('images/extracted/'+str(index+1)+'.png')
            roi        = makeSameSize (roi,savedImage)
            similar    = compareImages(roi,savedImage)
            
            if similar == False:

                cv2.imwrite('images/extracted/'+str(index)+'.png', roi)
                index = index - 1

    return index

def display(frame):

    cv2.imshow("Contours" , frame)

cap = cv2.VideoCapture(1);
while True:
        
    # Get Image
    _, frame = cap.read();
    index    = 12          # Total Number of Cards we are able to detect
    mask     = createMask(frame)
    contours = findContours(mask)
    
    # Should be creating and dealing with an array of card type objects 
    cardObj = card()
    cardList = []
    for i in range(index):
        a_card  = card()
        cardList.append(a_card) 

    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        x,y,w,h = cv2.boundingRect(cnt)

        if len(approx)==4 and perimeter > 400 and perimeter < 1400 and w > 120 and h > 180:

                drawBoundingBox(x,y,w,h,index,perimeter)
                roi=frame[y:y+h,x:x+w]
                index = saveImage(roi,index)
                        
    display(frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:  break;

cv2.destroyAllWindows()
cap.release()

