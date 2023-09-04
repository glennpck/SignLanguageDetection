import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

capture = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
offset = 20
size = 300

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        canvas = np.ones((size, size, 3),np.uint8) * 255

        crop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        cropShape = crop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            const = size/h
            wCal = math.ceil(const * w)

            resize = cv2.resize(crop, (wCal, size))
            resizeShape = resize.shape
            wGap = math.ceil((size-wCal)/2)

            canvas[:, wGap:wCal + wGap] = resize

        else:
            const = size/w
            hCal = math.ceil(const * h)

            resize = cv2.resize(crop, (size, hCal))
            resizeShape = resize.shape
            hGap = math.ceil((size-hCal)/2)

            canvas[hGap:hCal + hGap, :] = resize


        cv2.imshow("ImageCrop", crop)
        cv2.imshow("Canvas", canvas)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

