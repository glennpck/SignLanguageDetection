import cv2
import mediapipe as mp
import pickle
import numpy as np

model_dict = pickle.load(open('./src/model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hand_process = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0 : 'B', 1 : 'D', 2 : 'E', 3 : 'F'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = mp_hand_process.process(frame_rgb)

    if results.multi_hand_landmarks:

        for result_landmark in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                result_landmark,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for result_landmark in results.multi_hand_landmarks:

            for i in range(len(result_landmark.landmark)):
                x = result_landmark.landmark[i].x
                y = result_landmark.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)


        x1 = int(min(x_) * w) - 10
        y1 = int(min(y_) * h) - 10

        x2 = int(max(x_) * w) - 10
        y2 = int(max(y_) * h) - 10
        
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()