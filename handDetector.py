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

"""
In 
"""

class HandDetector:
    def __init__(self):
        self.save_interval = 1  # frame
        self.output_folder = "./gestures/test/"
        self.gestures = []
        self.windows = 10
        self.vote = 0.7
        self.model = load_model('models/CNN(200).h5')
        # self.model = load_model('models/weights_MLP_1100.h5')
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
        self.mp_drawing = mp.solutions.drawing_utils
        self.read_gesture_file()
        self.capture_image()
        

    def detect_hand_landmarks(self, image):
        results = self.hands.process(image)
        return results

    def draw_landmarks_on_image(self, image, results):
        landmark_coords = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    landmark_x = landmark.x * image.shape[1]
                    landmark_y = landmark.y * image.shape[0]
                    landmark_coords.append((landmark_x, landmark_y))

        return image, landmark_coords
    

    def capture_image(self):
        # mp_hands = mp.solutions.hands
        # detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
        cap = cv2.VideoCapture(0)
        frame_count = 0
        os.makedirs(self.output_folder, exist_ok=True)
        ten_y = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # cv2.imshow("Frame", frame)
            if frame_count % self.save_interval == 0:
                time1 = time.time()

                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect hand landmarks in the image
                detection_result = self.detector.process(frame_rgb)

                if detection_result.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)

                    # Write the frame with landmarks to disk
                    cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)

                    # Display the frame with landmarks
                    # cv2.imshow("Annotated Frame", annotated_image)

                    # Pre-process landmark coordinates for prediction
                    # process_landmark = self.pre_process_landmark(np.array(landmark_coords))
                    process_landmark = self.pre_process_landmark(landmark_coords, 'Right')

                    # Predict gesture
                    for point in process_landmark:
                        process_landmark_array = np.array(point).reshape(1, 21, 2)
                        time2 = time.time()
                        prediction = self.model.predict(process_landmark_array, verbose=0)
                        # print('only predict: {:2.2f} s'.format(time.time() - time2))
                        prediction_index = np.argmax(prediction)
                        print(self.gestures[prediction_index])
                        ten_y.append(prediction_index)
                        if len(ten_y) == self.windows:
                            most_action = max(set(ten_y), key=ten_y.count)
                            action = ten_y.count(most_action)
                            if self.vote <= action / self.windows and most_action != 9:
                                print(Fore.LIGHTCYAN_EX + f"DO THE ACTION {self.gestures[most_action]}")
                                print(Style.RESET_ALL)
                            ten_y.pop(0)

                # Display the original frame
                cv2.imshow("Frame", frame)
                # print('all things: {:2.2f} s'.format(time.time() - time1))

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



    def read_gesture_file(self):
        filename = './dataset/keypoint_classifier_label.csv'
        df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
        self.gestures = df.values.tolist()





    def pre_process_landmark(self, landmarks, chosen_hand):
        print(landmarks)
        processed_landmarks = []
        
        # Check if both hands are detected
        if len(landmarks) == 2:
            left_hand_type = "Left" if landmarks[0].landmark[0].x < landmarks[1].landmark[0].x else "Right"
            right_hand_type = "Right" if landmarks[0].landmark[0].x < landmarks[1].landmark[0].x else "Left"
        else:
            left_hand_type = "Left" if landmarks[0].landmark[0].x < 0.5 else "Right"
            right_hand_type = "Right" if landmarks[0].landmark[0].x < 0.5 else "Left"
        
        for hand_landmarks in landmarks:
            # Determine the hand type
            hand_type = "Right" if hand_landmarks.landmark[0].x < hand_landmarks.landmark[17].x else "Left"
            
            # Check if it's the chosen hand or if both hands are detected
            if (hand_type == chosen_hand and hand_type == right_hand_type) or chosen_hand == "both":
                # Normalize the landmarks
                base_x, base_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                max_value = max(max(abs(point.x - base_x), abs(point.y - base_y)) for point in hand_landmarks.landmark)

                normalized_landmarks = [[(point.x - base_x) / max_value, (point.y - base_y) / max_value] for point in hand_landmarks.landmark]
                processed_landmarks.append(normalized_landmarks)
        
        return processed_landmarks

    # def pre_process_landmark(self, landmark):
    #     # print(landmark, len(landmark[0]),len(landmark), '\n\n')
    #     hands = []
    #     if len(landmark) > 21:
    #         hands.append(landmark[0:21])
    #         hands.append(landmark[21:])
    #     else:
    #         hands.append(landmark[0:21])

    #     # print(12345678,hands)
    #     landmark_list = np.array(hands)


    #     final = []
    #     # print('12345678,', landmark_list)

    #     for landmark_list_hand in landmark_list:


    #         temp_landmark_list = copy.deepcopy(landmark_list_hand)


    #         base_x, base_y = 0, 0
    #         for index, landmark_point in enumerate(temp_landmark_list):
    #             if index == 0:
    #                 base_x, base_y = landmark_point[0], landmark_point[1]

    #             temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
    #             temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y


    #         temp_landmark_list = list(
    #             itertools.chain.from_iterable(temp_landmark_list))
            
    #         if temp_landmark_list != [] :
    #             max_value = max(list(map(abs, temp_landmark_list)))

    #         def normalize_(n):
    #             return n / max_value
            
    #         temp_landmark_list = list(map(normalize_, temp_landmark_list))
    #         final.append(temp_landmark_list)

    #     # print('\n\n\n\n final', final)
    #     return final
                





