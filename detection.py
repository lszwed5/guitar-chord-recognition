import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import time
from pymediainfo import MediaInfo
import csv
import os
from PIL import Image


class Detection:
    def __init__(self, source, reflection=False, skip_frames=False):
        self.source = source
        self.size = self.get_video_size()
        self.reflection = reflection
        self.skip_frames = skip_frames
        self.FPS = None
        self.previous_time = 0
        self.classification = ''
        self.subframe = [[], []]

    def get_video_size(self):
        media_info = MediaInfo.parse(self.source)
        video_track = media_info.video_tracks[0]
        return video_track.width, video_track.height

    def resize(self, size):
        self.size = size

    def reflect_video(self, frame):
        return cv2.flip(frame, 1)

    def show(self, guitar=True, fingers=True):
        self.init_YOLO_model()
        self.init_MP_gesture_model()
        self.init_MP_model()

        video_capture = cv2.VideoCapture(self.source)

        i = 0
        while True:
            i += 1
            success, frame = video_capture.read()
            if not success:
                print("End of video")
                break

            if self.reflection:
                frame = self.reflect_video(frame)

            if self.skip_frames:
                if not i % self.skip_frames:
                    continue

            frame = cv2.resize(frame, self.size)

            if len(self.subframe[0]):
                print('X:', self.subframe[0])
                print('Y:', self.subframe[1])
                subframe = frame[self.subframe[1][0]:self.subframe[1][1],
                                 self.subframe[0][0]:self.subframe[0][1]]
            else:
                subframe = frame

            # self.mark_gesture(frame)
            self.mark_gesture(np.array(subframe))
            self.mark_fingers(frame, fingers)
            if guitar:
                self.mark_guitar(frame)
            self.display_classification(frame, self.classification)
            self.display_FPS(frame)

            cv2.imshow("Imaage", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def init_YOLO_model(self):
        self.guitar_model = YOLO('models/best_run1.pt')
        data = open("coco.txt", "r").read()
        self.class_list = data.split("\n")

    def init_MP_model(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5)
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def init_MP_gesture_model(self):
        model_file = open('models/C_G_D_model_better.task', "rb")
        model_data = model_file.read()
        model_file.close()

        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def mark_gesture(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognition_result = self.recognizer.recognize(mp_image)
        top_gesture = recognition_result.gestures
        if len(top_gesture):
            if top_gesture[0][0].score > 0.5:
                print("Classification:", top_gesture[0][0].category_name)
                if top_gesture[0][0].category_name == '':
                    self.classification = "-"
                else:
                    self.classification = top_gesture[0][0].category_name

    def mark_fingers(self, frame, display):
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.multi_hand_landmarks:
            for hand_id, hand_landmark in enumerate(
                    result.multi_hand_landmarks):
                if result.multi_handedness[0].classification[0].score < 0.95:
                    continue
                if hand_id == 0:
                    self.get_subframe_position(hand_landmark.landmark)
                    if display:
                        self.mp_drawing_utils.draw_landmarks(
                            frame, hand_landmark,
                            self.mp_hands.HAND_CONNECTIONS)

    def save_positions(self, data):
        output_file = 'data/g.csv'
        with open(output_file, 'a', newline="") as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(data)
        print("done")

    def get_subframe_position(self, hand_landmark):
        scale = 0.2
        X, Y = [], []
        for finger_id, landmark in enumerate(hand_landmark):
            x, y = int(landmark.x * self.size[0]), int(
                landmark.y * self.size[1])
            X.append(x)
            Y.append(y)
        self.subframe[0] = [max(int(min(X) - self.size[0] * scale), 0),
                            min(int(max(X) + self.size[0] * scale),
                                self.size[0])]
        self.subframe[1] = [max(int(min(Y) - self.size[1] * scale), 0),
                            min(int(max(Y) + self.size[1] * scale),
                                self.size[1])]

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

    def display_classification(self, frame, text):
        width = int(self.size[0])
        height = int(self.size[1])

        cv2.rectangle(
            frame,
            (width // 2 - 50, height),
            (width // 2 + 50, height - 80),
            (50, 50, 50),
            -1
        )

        cv2.putText(
            frame,
            text,
            (width // 2 - 25, height - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (255, 255, 255),
            2
        )
