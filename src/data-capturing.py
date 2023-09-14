import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp
import pickle

capture = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1, modelComplexity=0)
offset = 20
size = 500

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hand_process = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


path = "img/A"
counter = 0
save_counter = 0
data = []
labels = []
label = 0
key = ""


while True:
    success, img = capture.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        data_aux = []
        hand = hands[0]
        x, y, w, h = hand['bbox']        

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        results = mp_hand_process.process(img_rgb)

        if results.multi_hand_landmarks:

            for result_landmark in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    img,
                    result_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(result_landmark.landmark)):
                    x = result_landmark.landmark[i].x
                    y = result_landmark.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            key = cv2.waitKey(1)
            if key == ord('s'):
                save_counter += 1
                data.append(data_aux)
                labels.append(label)        
                print("Data Saved = " + str(save_counter) + ", current label = " + str(label))

    cv2.imshow("Image", img)
    if key == ord('d'):
        label = 1
    elif key == ord('e'):
        label = 2
    elif key == ord('f'):
        label = 3
    elif key == ord('x'):
        break

print(data)
print(labels)
proceed = input("Proceed? [Y/N]: ")
if proceed.upper() == "Y":
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()