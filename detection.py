import cv2
from ultralytics import YOLO
import mediapipe as mp
import pandas as pd
import time
from pymediainfo import MediaInfo


class Detection:
    def __init__(self, source):
        self.source = source
        self.size = self.get_video_size()
        self.FPS = None
        self.previous_time = 0

    def get_video_size(self):
        media_info = MediaInfo.parse(self.source)
        video_track = media_info.video_tracks[0]
        return video_track.width, video_track.height

    def resize(self, size):
        self.size = size

    def show(self):
        self.init_YOLO_model()
        self.init_MP_model()

        video_capture = cv2.VideoCapture(self.source)

        while True:
            success, frame = video_capture.read()
            if not success:
                print("End of video")
                break

            frame = cv2.resize(frame, self.size)

            self.mark_fingers(frame)
            self.mark_guitar(frame)

            self.display_FPS(frame)

            cv2.imshow("Imaage", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


    def init_YOLO_model(self):
        self.guitar_model = YOLO('best_run1.pt')
        data = open("coco.txt", "r").read()
        self.class_list = data.split("\n")

    def init_MP_model(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def mark_fingers(self, frame):
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                self.mp_drawing_utils.draw_landmarks(frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

    def mark_guitar(self, frame):
        guitar_result = self.guitar_model.predict(frame)
        px = pd.DataFrame(guitar_result[0].boxes.data).astype("float")
        tmp = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = self.class_list[d]
            tmp.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c).title(), (int(x2), int(y2)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    def display_FPS(self, frame):
        current_time = time.time()
        self.FPS = int(1 / (current_time - self.previous_time))
        self.previous_time = current_time

        cv2.putText(
            frame,
            f'FPS: {self.FPS}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (9, 240, 5),
            2
        )
