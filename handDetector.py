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

class HandDetector:
    def __init__(self):

        self.gestures = []
        self.windows = 10
        self.vote = 0.7
        self.model = load_model('models/weights_CNN_my800.h5')
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
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
        # hand_detector = HandDetector()
        cap = cv2.VideoCapture(0)
        frame_count = 0
        save_interval = 20  # seconds
        output_folder = "./gestures/test/"
        os.makedirs(output_folder, exist_ok=True)



        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Display the frame
            cv2.imshow("Frame", frame)

            # Capture and save frame every save_interval seconds
            if frame_count % (save_interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
                # Detect hand landmarks
                detection_result = self.detect_hand_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                
                # Draw landmarks on the image and get coordinates
                annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                # # Save the annotated image
                # cv2.imwrite(f"{output_folder}annotated_frame{frame_count}.jpg", annotated_image)

                input_model = np.array(landmark_coords)
                process_landmark = self.pre_process_landmark(input_model)
                process_landmark = np.reshape(process_landmark, (-1, 2))


                # Make prediction
                if len(process_landmark) != 0:
                    process_landmark_array = np.array(process_landmark).reshape(1, 21, 2)
                    prediction = self.model.predict(process_landmark_array)
                    print('prediction:', frame_count, prediction)
                    prediction_index = np.argmax(prediction)
                    print("Index of highest value in prediction:", prediction_index, self.gestures[prediction_index])

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # return landmark_coords

    def read_gesture_file(self):
        filename = './dataset/keypoint_classifier_label.csv'
        df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
        self.gestures = df.values.tolist()


    def pre_process_landmark(self, landmark_list):
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
            
