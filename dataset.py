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
import urllib
import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import cv2
from collections import Counter
# from google.colab.patches import cv2_imshow



# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green




class Dataset:
    def __init__(self):
        self.capture_image()
        # self.infos()

    def pre_process_landmark(self, landmark):
        hands, final = [], []
        if len(landmark) > 21:
            hands.append(landmark[0:21])
            hands.append(landmark[21:])
        else:
            hands.append(landmark[0:21])
        landmark_list = np.array(hands)

        for landmark_list_hand in landmark_list:
            temp_landmark_list = copy.deepcopy(landmark_list_hand)
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
            final.append(temp_landmark_list)

        return final


    def draw_landmarks_on_image(self, image, results):
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            landmark_coords = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for landmark in hand_landmarks.landmark:
                        landmark_x = landmark.x * image.shape[1]
                        landmark_y = landmark.y * image.shape[0]
                        landmark_coords.append((landmark_x, landmark_y))
            return image, landmark_coords
    
    

    def capture_image(self):
        # Initialize MediaPipe Hand Landmarker
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

        # Initialize VideoCapture
        cap = cv2.VideoCapture(0)
        frame_count = 0
        save_interval = 1  # seconds
        output_folder = "./gestures/test/"
        os.makedirs(output_folder, exist_ok=True)

        detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Open CSV file
        with open('./dataset/right_hand.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # cv2.imshow("Frame", frame)

                if frame_count % save_interval == 0:
                    # Convert the frame to RGB format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert the RGB frame to MediaPipe's image format
                    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  
                    image.flags.writeable = False  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

                    # Detect hand landmarks in the image
                    detection_result = detector.process(image)

                    if detection_result.multi_hand_landmarks:
                        # Draw landmarks on the image
                        annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)

                        # Save the annotated image
                        cv2.imwrite(os.path.join(output_folder, f"annotated_frame_{frame_count}.jpg"), annotated_image)

                        # Extract coordinates from detection result
                        process_landmark = self.pre_process_landmark(np.array(landmark_coords))

                        # Check for key press
                        key = cv2.waitKey(1)
                        if key != -1:
                            if ord('0') <= key <= ord('9') and len(process_landmark) != 0:
                                for point in process_landmark:
                                    row_data = [chr(key)] + point[:42]
                                    csvwriter.writerow(row_data)
                                    print(f"Saved a list for {chr(key)}\n")

                cv2.imshow("Frame", frame)
                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the camera and close the CSV file
        cap.release()
        cv2.destroyAllWindows()


    def infos(self):
        counter = Counter()

        # Open the CSV file
        with open('./dataset/dataset_1hand(10).csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # Iterate through each row in the CSV file
            for row in csvreader:
                # Get the first index (the first value in each row)
                first_index = row[0]
                
                # Increment the counter for this first index
                counter[first_index] += 1

        # Print the count of each unique value in the first index
        for key, value in counter.items():
            print(f'Index {key}: {value} occurrences')

