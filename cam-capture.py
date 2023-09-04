import cv2
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        crop = img[y:y+h, x:x+w]
        cv2.imshow("ImageCrop", crop)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

