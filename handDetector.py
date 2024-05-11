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

    def draw_landmarks_on_image(self, image, results, chosen_hand):
        # myHands=[]
        # handsType=[]
        # # frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # # results=self.hands.process(frameRGB)
        # hands = {}
        # width=1280
        # height=720
        # if results.multi_hand_landmarks != None:
        #     #print(results.multi_handedness)
        #     for hand in results.multi_handedness:
        #         #print(hand)
        #         #print(hand.classification)
        #         #print(hand.classification[0])
        #         handType=hand.classification[0].label
        #         handsType.append(handType)
        #     for handLandMarks in results.multi_hand_landmarks:
        #         myHand=[]
        #         for landMark in handLandMarks.landmark:
        #             myHand.append((int(landMark.x*width),int(landMark.y*height)))
        #         myHands.append(myHand)

        # print(myHands,handsType)                

        # # return myHands,handsType

  


        hands = []
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
                        
                    # hands.append(landmark_coords)

        if chosen_hand in handsType:
            self.mp_drawing.draw_landmarks(
                image, results.multi_hand_landmarks[handsType.index(chosen_hand)], self.mp_hands.HAND_CONNECTIONS)

        # print(handsType, hands)
        # if len(hands) != len(handsType):
        #     print('hiiiii')
        #     print(handsType)
        #     print(hands)
        #     return 
        # If the chosen hand is found, return its landmarks, otherwise return an empty list
        if chosen_hand in handsType:
            # print(chosen_hand, handsType.index(chosen_hand),len(hands), hands, '\n')
            return image, landmark_coords[0:21]
        else:
            return image, []


        # if len(handsType) == 1:
        #     print('ONE HAND')
        #     if handsType[0] == chosen_hand:
        #         print('TRUE HAND')
        #         print(landmark_coords)
        #         return image, landmark_coords
        #     else:
        #         print('NOT TRUE HAND')
        #         return image, []
            
        # elif len(handsType) == 2:
        #     print('TWO HANDS')
        #     if handsType[0] == chosen_hand:
        #         print(landmark_coords[0:21], handsType[0])
        #         return image, landmark_coords[0:21]
        #     elif handsType[1] == chosen_hand:
        #         print(landmark_coords[21:], handsType[1])
        #         return image, landmark_coords[21:]






            

        # landmark_coords = np.array(landmark_coords)
        # chosen_hands = []

        # if len(landmark_coords) == 21:
        #     print('ONE HAND')
        #     # print(results.multi_handedness)
        #     results_dict = MessageToDict(results.multi_handedness[0])
        #     # print(results_dict['classification'][0]['label'])
        #     hand_type = results_dict['classification'][0]['label']
        #     if chosen_hand == hand_type:
        #         # landmark_coords 
        #         print('TRUE HAND')
        #         print(landmark_coords)
        #         return image, landmark_coords
        #     else:
        #         print('NOT TRUE HAND')
        #         return image, []

        # else:
        #     print('TWO HANDS')
        #     print(results.multi_handedness)
        #     print(landmark_coords)
        #     if chosen_hand == 'Left':
        #         print('LEFT')
        #         print(landmark_coords[21:])
        #         return image, landmark_coords[21:]
        #         # return image, landmark_coords[0:21]
        #     else:
        #         print('Right')
        #         print(landmark_coords[0:21])
        #         # return image, landmark_coords[21:]
        #         return image, landmark_coords[0:21]


        
        # if len(landmark_coords) > 21:
        #     hands.append(landmark_coords[0:21])
        #     hands.append(landmark_coords[21:])
        # else:
        #     hands.append(landmark_coords[0:21])


        # print(results.multi_handedness)
        

        # print(12345678,hands)
        # landmark_list = np.array(hands)
        # print(landmark_coords)

        # return image, landmark_coords
    

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
                # print(detection_result.multi_hand_landmarks)
                # print(detection_result.multi_handedness)

                if detection_result.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result, 'Right')
                    # self.draw_landmarks_on_image(frame, detection_result, 'Right')
                    # print(landmark_coords)

                    # Write the frame with landmarks to disk
                    cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)

                    # Display the frame with landmarks
                    # cv2.imshow("Annotated Frame", annotated_image)

                    # Pre-process landmark coordinates for prediction
                    process_landmark = self.pre_process_landmark(np.array(landmark_coords))
                    # print(process_landmark)
                    # process_landmark = self.pre_process_landmark(landmark_coords, 'Right')

                    # Predict gesture
                    # for point in process_landmark:
                    # print(process_landmark)
                    if process_landmark != []:

                        process_landmark_array = np.array(process_landmark).reshape(1, 21, 2)
                        time2 = time.time()
                        prediction = self.model.predict(process_landmark_array, verbose=0)
                        # print('only predict: {:2.2f} s'.format(time.time() - time2))
                        prediction_index = np.argmax(prediction)
                        print(self.gestures[prediction_index])
                        ten_y.append(prediction_index)
                        if len(ten_y) == self.windows:
                            most_action = max(set(ten_y), key=ten_y.count)
                            action = ten_y.count(most_action)
                            print(prediction_index)
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





    # def pre_process_landmark(self, landmarks, chosen_hand):
    #     print(len(landmarks))
    #     processed_landmarks = []

    #     if len(landmarks) == 1:  # Only one hand detected
    #         print('ONE HAND')
    #         hand_landmarks = landmarks[0].landmark
    #         # Determine the type of the detected hand
    #         hand_type = "Right" if hand_landmarks[0].x < 0.5 else "Left"

    #         if chosen_hand == "both" or chosen_hand == hand_type:
    #             # Normalize the landmarks
    #             base_x, base_y = hand_landmarks[0].x, hand_landmarks[0].y
    #             max_value = max(max(abs(point.x - base_x), abs(point.y - base_y)) for point in hand_landmarks)

    #             normalized_landmarks = [[(point.x - base_x) / max_value, (point.y - base_y) / max_value] for point in hand_landmarks]
    #             processed_landmarks.append(normalized_landmarks)

    #     elif len(landmarks) == 2:  # Two hands detected
    #         left_hand = landmarks[0].landmark if landmarks[0].landmark[0].x < landmarks[1].landmark[0].x else landmarks[1].landmark
    #         right_hand = landmarks[1].landmark if landmarks[0].landmark[0].x < landmarks[1].landmark[0].x else landmarks[0].landmark

    #         left_hand_type = "Left" if left_hand[0].x < 0.5 else "Right"
    #         right_hand_type = "Left" if right_hand[0].x < 0.5 else "Right"

    #         for hand_landmarks in [left_hand, right_hand]:
    #             hand_type = "Right" if hand_landmarks[0].x < hand_landmarks[17].x else "Left"
    #             if (chosen_hand == "both" or chosen_hand == hand_type) and ((chosen_hand == "right" and hand_type == right_hand_type) or (chosen_hand == "left" and hand_type == left_hand_type)):
    #                 base_x, base_y = hand_landmarks[0].x, hand_landmarks[0].y
    #                 max_value = max(max(abs(point.x - base_x), abs(point.y - base_y)) for point in hand_landmarks)

    #                 normalized_landmarks = [[(point.x - base_x) / max_value, (point.y - base_y) / max_value] for point in hand_landmarks]
    #                 processed_landmarks.append(normalized_landmarks)

    #     return processed_landmarks

    def pre_process_landmark(self, landmark):
        # # print(landmark, len(landmark[0]),len(landmark), '\n\n')
        # hands = []
        # if len(landmark) > 21:
        #     hands.append(landmark[0:21])
        #     hands.append(landmark[21:])
        # else:
        #     hands.append(landmark[0:21])

        # print(12345678,hands)
        landmark_list = np.array(landmark)
        # print(landmark_list)


        final = []
        # print('12345678,', landmark_list)

        # for landmark_list_hand in landmark_list:


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
        # final.append(temp_landmark_list)

        # print('\n\n\n\n final', final)
        return temp_landmark_list
                






