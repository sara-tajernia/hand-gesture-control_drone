import cv2
import mediapipe as mp
import os
from keras.models import load_model
import numpy as np
import csv
import pandas as pd
import copy
import itertools
from preprocess_data import Preprocess
from collections import Counter
from colorama import Fore, Back, Style
import time
from google.protobuf.json_format import MessageToDict
from control_drone import ControlDrone



class HandDetector:
    def __init__(self):
        self.save_interval = 1  # frame
        self.output_folder = "./gestures/test/"
        self.gestures = []
        self.windows = 10
        self.vote = 0.7
        self.model = load_model('models/CNN_left.h5')
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
        self.mp_drawing = mp.solutions.drawing_utils
        self.read_gesture_file()
        self.status = False
        self.drone = ControlDrone()
        self.capture_image()
        

    def detect_hand_landmarks(self, image):
        results = self.hands.process(image)
        return results

    def draw_landmarks_on_image(self, image, results, chosen_hand):
        handsType=[]
        if results.multi_hand_landmarks:
            landmark_coords = []
            for hand, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                handType = hand.classification[0].label
                handsType.append(handType)
                
                # Check if this hand is the chosen hand
                if handType == chosen_hand:
                    for landmark in hand_landmarks.landmark:
                        landmark_x = int(landmark.x * image.shape[1])
                        landmark_y = int(landmark.y * image.shape[0])
                        landmark_coords.append((landmark_x, landmark_y))

        if chosen_hand in handsType:
            self.mp_drawing.draw_landmarks(
                image, results.multi_hand_landmarks[handsType.index(chosen_hand)], self.mp_hands.HAND_CONNECTIONS)

        # If the chosen hand is found, return its landmarks, otherwise return an empty list
        if chosen_hand in handsType:
            return image, landmark_coords[0:21]
        else:
            return image, []



    def capture_image(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        os.makedirs(self.output_folder, exist_ok=True)
        ten_y = []

        # while True:
        #     frame = self.drone.get_frame()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.save_interval == 0:
                time1 = time.time()

                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_result = self.detector.process(frame_rgb)
                text = "None"
                if detection_result.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result, 'Right')

                    # Write the frame with landmarks to disk
                    cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)
                    process_landmark = self.pre_process_landmark(np.array(landmark_coords))

                    if process_landmark != []:

                        process_landmark_array = np.array(process_landmark).reshape(1, 21, 2)
                        time2 = time.time()
                        prediction = self.model.predict(process_landmark_array, verbose=0)
                        # print('only predict: {:2.2f} s'.format(time.time() - time2))
                        prediction_index = np.argmax(prediction)
                        ten_y.append(prediction_index)
                        if len(ten_y) == self.windows:
                            most_action = max(set(ten_y), key=ten_y.count)
                            action = ten_y.count(most_action)
                            # print(prediction_index)
                            if self.vote <= action / self.windows and most_action != 9:
                                # if not self.status:      
                                    # if most_action == 0:       
                                        # self.status = True
                                        # print(Fore.LIGHTCYAN_EX + f"Starting {self.gestures[most_action]}")
                                    text = self.gestures[most_action][0]
                                    self.drone.follow_order(most_action)
                                    # else:
                                        # # print(Fore.LIGHTCYAN_EX + f"DO THE ACTION {self.gestures[most_action]}")
                                        # text = self.gestures[most_action][0]
                                        # self.drone.follow_order(most_action)

                            ten_y.pop(0)

                # Display the original frame
                # text = "Hello, OpenCV!"  
                font = cv2.FONT_HERSHEY_SIMPLEX  
                fontScale = 1 
                color = (255, 255, 255) 
                thickness = 2  

                # Put the text on top of the frame
                cv2.putText(frame, text, (10, 50), font, fontScale, color, thickness)
                cv2.imshow("Frame", frame)
                # print('all things: {:2.2f} s'.format(time.time() - time1))

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.drone.land()
                break

        cap.release()
        cv2.destroyAllWindows()



    def read_gesture_file(self):
        filename = './dataset/keypoint_classifier_label.csv'
        df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
        self.gestures = df.values.tolist()


    def pre_process_landmark(self, landmark):
        landmark_list = np.array(landmark)
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y


        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        
        if temp_landmark_list != [] :
            max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value
        
        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list
                