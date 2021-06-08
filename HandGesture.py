import time
import numpy as np
import cv2
import math
import pyttsx3 as pt
result=''
# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_frame = frame[100:300, 100:300]
    gray=cv2.cvtColor(crop_frame,cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Change color-space from BGR -> HSV
    _,thresh1=cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours,hie=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,\
                                  cv2.CHAIN_APPROX_NONE)


    contour = max(contours, key=lambda x: cv2.contourArea(x))
    # Create a binary image with where white will be skin colors and rest is black
    #mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Create bounding rectangle around the contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (0, 0, 255), 0)

    # Find convex hull
    hull = cv2.convexHull(contour)

    # Draw contour
    drawing = np.zeros(crop_frame.shape, np.uint8)
    cv2.drawContours(drawing, [contour], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    # Find convexity defects
    hull = cv2.convexHull(contour, returnPoints=False)
    defects=cv2.convexityDefects(contour, hull)


    # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
    # tips) for all defects
    count_defects = 0

    cv2.drawContours(thresh1,contours,-1,(0,255,0),3)

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
        # if angle > 90 draw a circle at the far point
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_frame, far, 1, [0, 0, 255], -1)
        cv2.line(crop_frame, start, end, [0, 255, 0], 2)

    if count_defects == 0:
        cv2.putText(frame, "Best of Luck", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
        result='Resultant   HAND   gesture   displayed   by   the   person  is   Best of Luck'

    elif count_defects == 1:
        cv2.putText(frame, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        result='Resultant HAND gesture displayed by the person is Peace'

    elif count_defects == 2:
        cv2.putText(frame, "Nice", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        result='Resultant HAND gesture displayed by the person is Nice'

    elif count_defects == 3:
        cv2.putText(frame, "I need four things", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        result='Resultant HAND gesture displayed by the person is Four'

    elif count_defects == 4:
        cv2.putText(frame, "Wait", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        result='Resultant HAND gesture displayed by the person is Wait'

        # Print number of fingers
    else:
        cv2.putText(frame, "PUT YOUR HAND IN FRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_frame))
    cv2.imshow('Contours', all_image)



    if cv2.waitKey(1) == ord('q'):
        break




print(result)
speaker=pt.init()
speaker.say(result)
time.sleep(2)
speaker.runAndWait()
capture.release()
cv2.destroyAllWindows()