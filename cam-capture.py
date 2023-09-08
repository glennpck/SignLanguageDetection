import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import mediapipe as mp

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


while True:
    success, img = capture.read()
    hands, img = detector.findHands(img, draw=False)
    if hands:
        data_aux = []
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

# ============== TAMPERING WITH CANVAS ==================            

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
        results = mp_hand_process.process(canvas_rgb)

        if results.multi_hand_landmarks:

            for result_landmark in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    canvas_rgb,
                    result_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(result_landmark.landmark)):
                    x = result_landmark.landmark[i].x
                    y = result_landmark.landmark[i].x
                    data_aux.append(x)
                    data_aux.append(y)

            save = cv2.waitKey(1)
            if save == ord('s'):
                save_counter += 1
                data.append(data_aux)
                labels.append("A")        
                print("Data Saved = " + str(save_counter))    
    
        cv2.imshow("Canvas", canvas_rgb)

#========================================================
        
        cv2.imshow("ImageCrop", crop)
        #cv2.imshow("Canvas", canvas)


    cv2.imshow("Image", img)

print(data)
print(labels)



    
    # key = cv2.waitKey(1)
    # if key == ord("s"):
    #     counter += 1
    #     cv2.imwrite(f'{path}/Image_{time.time()}.png', canvas)
    #     print("Saved Images: " + str(counter))

