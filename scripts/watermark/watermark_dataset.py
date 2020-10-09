# USAGE
# python watermark_dataset.py --watermark pyimagesearch_watermark.png --input input --output output

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

font = cv2.FONT_HERSHEY_SIMPLEX
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

watermark = cv2.imread("./pyimagesearch_watermark.png", cv2.IMREAD_UNCHANGED)
(wH, wW) = watermark.shape[:2]

(B, G, R, A) = cv2.split(watermark)
B = cv2.bitwise_and(B, B, mask=A)
G = cv2.bitwise_and(G, G, mask=A)
R = cv2.bitwise_and(R, R, mask=A)
watermark = cv2.merge([B, G, R, A])

import time

start = time.time()
while(True):
	ret, image = cap.read()
	(h, w) = image.shape[:2]
	image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

	overlay = np.zeros((h, w, 4), dtype="uint8")
	overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark

	# blend the two images together using transparent overlays
	output = image.copy()
	cv2.addWeighted(overlay, 0.55, output, 1.0, 0, output)
	delta = time.time() - start
	res = "DUREE :" + str(round(delta,2))
	cv2.putText(output, res ,(10,300), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the resulting frame
	cv2.imshow('frame',output)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
