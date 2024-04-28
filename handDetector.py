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


"""
In 
"""

class HandDetector:
    def __init__(self):
        self.save_interval = 3  # frame
        self.output_folder = "./gestures/test/"
        self.gestures = []
        self.windows = 10
        self.vote = 0.7
        self.model = load_model('models/weights_CNN_new.h5')
        # self.model = load_model('models/weights_MLP_1100.h5')
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

        # print('12345678', landmark_coords)
        return image, landmark_coords
    
    def pre_process_landmark(self, landmark):
        # print(landmark, len(landmark[0]),len(landmark), '\n\n')
        hands = []
        if len(landmark) > 21:
            hands.append(landmark[0:21])
            hands.append(landmark[21:])
        else:
            hands.append(landmark[0:21])

        # print(12345678,hands)
        landmark_list = np.array(hands)


        final = []
        # print('12345678,', landmark_list)

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

        # print('\n\n\n\n final', final)
        return final

    def capture_image(self):
        mp_hands = mp.solutions.hands
        detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        frame_count = 0
        os.makedirs(self.output_folder, exist_ok=True)
        ten_y = []


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Display the frame
            cv2.imshow("Frame", frame)

            if frame_count % self.save_interval == 0:
              # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert the RGB frame to MediaPipe's image format
                image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR format for MediaPipe
                image.flags.writeable = False  # Set writable flag to False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB

                # Detect hand landmarks in the image
                detection_result = detector.process(image)
                # print(detection_result.multi_hand_landmarks)
                # print(detection_result.multi_handedness)

                # if detection_result.multi_hand_landmarks:
                #     classification_list = detection_result.multi_handedness[0]
                #     if len(classification_list.classification) == 1:
                #         classification = classification_list.classification[0]
                #         print(classification.label)

                        # if classification.label == "Right":
                        #     print('right')
                        #     annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                        #     cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)
                        #     process_landmark = self.pre_process_landmark(np.array(landmark_coords))   
                        #     print(process_landmark, '\n')  

                        # else:
                        #     print('left', detection_result)
                        #     annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                        #     cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)
                        #     process_landmark = self.pre_process_landmark(np.array(landmark_coords))   
                        #     print(process_landmark, '\n')  

                if detection_result.multi_hand_landmarks:                
                    annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                    cv2.imwrite(os.path.join(f"./gestures/test/annotated_frame_{frame_count}.jpg"), annotated_image)
                    process_landmark = self.pre_process_landmark(np.array(landmark_coords))   
                    # print(process_landmark)  

                    for point in process_landmark:
                        # print(point)
                        if len(process_landmark) >1:
                            print(Fore.BLUE + "2 HANDS")
                            print(Style.RESET_ALL)
                        process_landmark_array = np.array(point).reshape(1, 21, 2)
                        prediction = self.model.predict(process_landmark_array)
                        prediction_index = np.argmax(prediction)
                        # print("Index of highest value in prediction:",prediction, prediction_index, self.gestures[prediction_index])
                        ten_y.append(prediction_index)
                        if len(ten_y) == self.windows:
                            most_action = max(set(ten_y), key = ten_y.count)
                            action = ten_y.count(most_action)
                            
                            if self.vote <= action/self.windows:
                                # print('*********',ten_y, action)
                                print(Fore.RED + f"DO THE ACTION {self.gestures[most_action]}", len(process_landmark))
                                print(Style.RESET_ALL)
                            ten_y.pop(0)


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


    def pre_process_landmark(self, landmark):
        # print(landmark, len(landmark[0]),len(landmark), '\n\n')
        hands = []
        if len(landmark) > 21:
            hands.append(landmark[0:21])
            hands.append(landmark[21:])
        else:
            hands.append(landmark[0:21])

        # print(12345678,hands)
        landmark_list = np.array(hands)


        final = []
        # print('12345678,', landmark_list)

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

        # print('\n\n\n\n final', final)
        return final
                









# import cv2
# import mediapipe as mp
# import os
# from keras.models import load_model
# import numpy as np
# import csv
# import pandas as pd
# import copy
# import itertools
# from preprocess_data import Preprocess
# from collections import Counter
# from colorama import Fore, Back, Style


# """
# In 
# """

# class HandDetector:
#     def __init__(self):
#         self.save_interval = 3  # frame
#         self.output_folder = "./gestures/test/"
#         self.gestures = []
#         self.windows = 10
#         self.vote = 0.7
#         self.model = load_model('models/weights_CNN_my1100.h5')
#         # self.model = load_model('models/weights_MLP_1100.h5')
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(max_num_hands=1)
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.read_gesture_file()
#         self.capture_image()
        

#     def detect_hand_landmarks(self, image):
#         results = self.hands.process(image)
#         return results

#     def draw_landmarks_on_image(self, image, results):
#         landmark_coords = []
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.mp_drawing.draw_landmarks(
#                     image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
#                 for landmark in hand_landmarks.landmark:
#                     landmark_x = landmark.x * image.shape[1]
#                     landmark_y = landmark.y * image.shape[0]
#                     landmark_coords.append((landmark_x, landmark_y))

#         print('12345678', landmark_coords)
#         return image, landmark_coords

#     def capture_image(self):
#         cap = cv2.VideoCapture(0)
#         frame_count = 0
#         os.makedirs(self.output_folder, exist_ok=True)
#         ten_y = []


#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1

#             # Display the frame
#             cv2.imshow("Frame", frame)

#             # print('byyy',frame_count,  self.save_interval, cap.get(cv2.CAP_PROP_FPS))

            

#             # Capture and save frame every save_interval seconds
#             # if frame_count % (self.save_interval * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
#             if frame_count % self.save_interval == 0:
#                 # Detect hand landmarks
#                 detection_result = self.detect_hand_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                
#                 # Draw landmarks on the image and get coordinates
#                 annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
#                 # # Save the annotated image
#                 cv2.imwrite(f"{self.output_folder}annotated_frame{frame_count}.jpg", annotated_image)

#                 input_model = np.array(landmark_coords)
#                 process_landmark = self.pre_process_landmark(input_model)
#                 process_landmark = np.reshape(process_landmark, (-1, 2))



#                 # Make prediction
#                 if len(process_landmark) != 0:
#                     process_landmark_array = np.array(process_landmark).reshape(1, 21, 2)
#                     prediction = self.model.predict(process_landmark_array)
#                     # print('prediction:', frame_count, prediction)
#                     prediction_index = np.argmax(prediction)
#                     print("Index of highest value in prediction:", prediction_index, self.gestures[prediction_index])

#                     ten_y.append(prediction_index)
#                     # print(ten_y)
#                     if len(ten_y) == self.windows:
#                         most_action = max(set(ten_y), key = ten_y.count)
#                         action = ten_y.count(most_action)
                        
#                         # print(2,ten_y, self.vote, action/self.windows, frame_count)
#                         if self.vote <= action/self.windows:
#                             # print('*********',ten_y, action)
#                             print(Fore.RED + f"DO THE ACTION {self.gestures[most_action]}")
#                             print(Style.RESET_ALL)
#                         ten_y.pop(0)

#             # Exit when 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#         # return landmark_coords

#     def read_gesture_file(self):
#         filename = './dataset/keypoint_classifier_label.csv'
#         df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
#         self.gestures = df.values.tolist()


#     def pre_process_landmark(self, landmark_list):
#         print('land,ark', landmark_list)
#         temp_landmark_list = copy.deepcopy(landmark_list)
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
#         return temp_landmark_list
            

