import cv2
import mediapipe as mp
import os
from keras.models import load_model
import numpy as np
import csv
import pandas as pd
import copy
import itertools
# import msvcrt
# import keyboard


class Dataset:
    def __init__(self):

        self.gestures = []
        self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands()
        self.hands = self.mp_hands.Hands(max_num_hands=1)  # Set max_num_hands to 1
        self.mp_drawing = mp.solutions.drawing_utils
        self.capture_image()

        
        # self.model = model

        

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
        print('landmark_coords' ,landmark_coords)
        return image, landmark_coords

    def capture_image(self):
        # hand_detector = HandDetector()
        cap = cv2.VideoCapture(0)
        frame_count = 0
        save_interval = 1  # seconds
        output_folder = "./gestures/test/"
        os.makedirs(output_folder, exist_ok=True)


        # model=load_model('models/weights_CNN_100.h5')

        with open('./dataset/my_dataset_new.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Display the frame
                cv2.imshow("Frame", frame)

                if frame_count % save_interval == 0:
                    detection_result = self.detect_hand_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                    cv2.imwrite(f"{output_folder}annotated_frame{frame_count}.jpg", annotated_image)

                    input_model = np.array(landmark_coords)
                    process_landmark = self.pre_process_landmark(input_model)
                    # print(2345678, type(process_landmark))

                    # Check for key press
                    key = cv2.waitKey(1)
                    if key != -1:
                        if ord('0') <= key <= ord('9') and len(process_landmark) != 0 :
                            row_data = [chr(key)] + process_landmark[:42]
                            csvwriter.writerow(row_data)
                            print(f"save a list for {chr(key)}")
                            # break

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the camera and close the CSV file
            cap.release()
            cv2.destroyAllWindows()




        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

        #     frame_count += 1

        #     # Display the frame
        #     cv2.imshow("Frame", frame)

        #     # Capture and save frame every save_interval seconds
        #     if frame_count % (save_interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
        #         # Detect hand landmarks
        #         detection_result = self.detect_hand_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #         # print(f'detection_result {detection_result} \n')

        #         annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)

        #         # print(f'landmark_coords {landmark_coords} \n')
        #         # print(f'frame_count {frame_count} \n')
                
        #         cv2.imwrite(f"{output_folder}annotated_frame{frame_count}.jpg", annotated_image)

        #         input_model = np.array(landmark_coords)
        #         # print(f'input_model {input_model} \n')

        #         process_landmark = self.pre_process_landmark(input_model)
        #         # print(f'process_landmark {process_landmark} \n\n\n')
        #         print(2345678, process_landmark)

        #     key = cv2.waitKey(1)
        #     if key != -1:
        #          if ord('0') <= key <= ord('9'):
        #             print(f"You pressed the number: {chr(key)}")


        #     # Exit when 'q' is pressed
        #     if key & 0xFF == ord('q'):
        #         break

        # cap.release()
        # cv2.destroyAllWindows()

        # # return landmark_coords



    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # 相対座標に変換
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # 1次元リストに変換
        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))
        
        # print('temp_landmark_list1', temp_landmark_list)

        # 正規化
        if temp_landmark_list != [] :
            max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value
        

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        # print('temp_landmark_list3', temp_landmark_list)

        return temp_landmark_list
            
