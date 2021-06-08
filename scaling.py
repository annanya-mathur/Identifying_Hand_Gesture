import cv2
cap=cv2.VideoCapture(0)
object_detector=cv2.createBackgroundSubtractorMOG2()
while True:
    r,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Blurring

    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Object Detection
    mask=object_detector.apply(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('Gray Scale',gray)
    cv2.imshow('Gaussian Blur ',blur)
    cv2.imshow('Object Detection',mask)



    if cv2.waitKey(1) ==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()