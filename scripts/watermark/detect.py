import cv2
import imutils
import time
import numpy as np
import beepy
from tkinter import Tk
lower_range = (29, 86, 6)
upper_range = (64, 255, 255)
font = cv2.FONT_HERSHEY_SIMPLEX

#lower_range = (0, 0, 0)
#upper_range = (10, 10, 10)

import random

xg = 90
yg = 90

count = 0

vs = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)
time.sleep(2.0)



start = time.time()
while True:
    _, frame = vs.read()

    if frame is None:
        break
    delta = time.time() - start
    res = "DUREE :" + str(round(delta,2))
    cv2.namedWindow ('Frame', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty ('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    width, height = frame.shape[:2]
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None



    print(xg)
    print(yg)
    center_coordinates_g = (xg, yg)
    # Radius of circle
    radius_g = 20
    # Blue color in BGR
    color_g = (255, 0, 0)
    # Line thickness of 2 px
    thickness_g = 2
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    cv2.circle(frame, center_coordinates_g, radius_g, color_g, thickness_g)


    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)


        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 5)
            cv2.imwrite("circled_frame.png", cv2.resize(frame, (int(height / 2), int(width / 2))))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if  abs(xg - x) < 20 and abs(yg-y) < 20:
                        if count < 10:
                            count +=1
                            xg = random.randint(1, height)
                            yg = random.randint(1, width)
                            Tk().bell()

    if count < 10:
        cv2.putText(frame, str(count) ,(500,50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, res ,(50,450), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        final = res
    else     :
        cv2.putText(frame, str(count) ,(500,50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Bravo" ,(50,350), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, final ,(50,450), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Frame", frame)
        time.sleep(5)
        count = 0
        start = time.time()

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
