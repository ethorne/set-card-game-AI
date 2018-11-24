import cv2
import timeit
from card import Card

def main():
    for i in range(1,15):
        cardName = 'images/extracted/' + str(i) + '.png';
        cardImage = cv2.imread(cardName)
        if cardImage is None:
            print 'Could not read image at ' + cardName
            continue
        print cardName
        start = timeit.default_timer()
        setCard = Card(cardImage)
        stop = timeit.default_timer()
        print(setCard)
        print "\nTIME ELAPSED : ", (stop - start), " s"
        print "\n------------------------------\n"

        windowName = 'image #' + str(i)
        cv2.imshow(windowName, cardImage)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()


main()
