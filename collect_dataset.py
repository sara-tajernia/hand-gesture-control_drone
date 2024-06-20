import os
import cv2
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp

from collections import Counter


class Dataset:
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.capture_image()

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
        mp_hands = mp.solutions.hands
        cap = cv2.VideoCapture(0)
        frame_count = 0
        save_interval = 1  
        output_folder = "./gestures/test/"
        os.makedirs(output_folder, exist_ok=True)

        detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        with open(self.path_dataset, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % save_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  
                    image.flags.writeable = False  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                    detection_result = detector.process(image)

                    if detection_result.multi_hand_landmarks:
                        annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                        cv2.imwrite(os.path.join(output_folder, f"annotated_frame_{frame_count}.jpg"), annotated_image)
                        process_landmark = self.pre_process_landmark(np.array(landmark_coords))

                        # Check for key press
                        key = cv2.waitKey(1)
                        if key != -1:
                            if ord('0') <= key <= ord('9') and len(process_landmark) != 0:
                                for point in process_landmark:
                                    row_data = [chr(key)] + point[:42]
                                    csvwriter.writerow(row_data)
                                    print(f"Save gesture {chr(key)}\n")

                cv2.imshow("Frame", frame)
                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


    def infos(self):
        counter = Counter()
        with open('./dataset/dataset_1hand(10).csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                first_index = row[0]
                counter[first_index] += 1

        for key, value in counter.items():
            print(f'Index {key}: {value} occurrences')

