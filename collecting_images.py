import cv2
import os
import time

Images_path='testing/'



cap= cv2.VideoCapture(0)
object_detector= cv2.createBackgroundSubtractorMOG2()
i=0
while (True):

    ret, frame = cap.read()
    mask=object_detector.apply(frame)
    imgname = os.path.join(Images_path + '.' + '{}.jpg'.format(str(i)))
    i+=1
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)

    time.sleep(5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap=cv2.destroyAllWindows()