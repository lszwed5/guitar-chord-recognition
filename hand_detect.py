import cv2
from ultralytics import YOLO
import mediapipe as mp
from tracker import Tracker
import pandas as pd
import time
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

guitar_model = YOLO('best_run1.pt')
data = open("coco.txt", "r").read()
class_list = data.split("\n")

video_capture = cv2.VideoCapture('videos/czarna_piesn.mp4')

# Create a hand landmarker with video mode
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing_utils = mp.solutions.drawing_utils

tracker = Tracker()
previous_time = 0
while True:
    # Read the next frame
    success, frame = video_capture.read()
    if not success:
        print("End of video")
        break

    frame = cv2.resize(frame, (1000, 500))
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    guitar_result = guitar_model.predict(frame)
    px = pd.DataFrame(guitar_result[0].boxes.data).astype("float")

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_drawing_utils.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    tmp = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        tmp.append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(c).title(), (int(x2), int(y2)),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    current_time = time.time()
    FPS = int(1/(current_time - previous_time))
    previous_time = current_time

    cv2.putText(
        frame,
        f'FPS: {FPS}',
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (9, 240, 5),
        2
    )
    cv2.imshow("Imaage", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

