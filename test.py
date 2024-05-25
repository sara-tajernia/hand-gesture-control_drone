import cv2
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




mp_drawing = mp.solutions.drawing_utils
model = load_model('models/CNN(200).h5')
mp_hands = mp.solutions.hands
detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)


def frame_video():
  vidcap = cv2.VideoCapture('vote.mov')
  success,image = vidcap.read()
  count = 1
  while success:
    cv2.imwrite("vote/frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

def write_pre(prediction, path):
    with open(path, 'a') as csvfile:
      csvfile.write(str(prediction))
      csvfile.write('\n')

def draw_landmarks_on_image(image, results, chosen_hand):

  hands = []
  handsType=[]
  if results.multi_hand_landmarks:
      landmark_coords = []
      for hand, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
          handType = hand.classification[0].label
          handsType.append(handType)
          
          if handType == chosen_hand:
              for landmark in hand_landmarks.landmark:
                  landmark_x = int(landmark.x * image.shape[1])
                  landmark_y = int(landmark.y * image.shape[0])
                  landmark_coords.append((landmark_x, landmark_y))
                  

  if chosen_hand in handsType:
      mp_drawing.draw_landmarks(
          image, results.multi_hand_landmarks[handsType.index(chosen_hand)], mp_hands.HAND_CONNECTIONS)

  if chosen_hand in handsType:
      return image, landmark_coords[0:21]
  else:
      return image, []



def pre_process_landmark(landmark):
        
        # print(12345678,hands)
        landmark_list = np.array(landmark)
        final = []
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
                




def capture_image():
  
    ten_y = []
    save_interval = 1  # frame
    output_folder = "./gestures/test/"
    gestures = []
    filename = './dataset/keypoint_classifier_label.csv'
    df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
    gestures = df.values.tolist()
    windows = 10
    vote = 0.7
    vidcap = cv2.VideoCapture('vote.mov')
    success, frame = vidcap.read()
    frame_count = 0
    while success:
      # cv2.imwrite("vote/frame%d.jpg" % frame_count, frame) 
      success,frame = vidcap.read()
      frame_count += 1
      # frame = image
      if frame_count % save_interval == 0:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks in the image
        detection_result = detector.process(frame_rgb)

        if detection_result.multi_hand_landmarks:
            # Draw landmarks on the frame
            annotated_image, landmark_coords = draw_landmarks_on_image(frame, detection_result, 'Left')
            cv2.imwrite(os.path.join(f"vote/annotated_frame_{frame_count}.jpg"), annotated_image)
            process_landmark = pre_process_landmark(np.array(landmark_coords))
            if process_landmark != []:

                process_landmark_array = np.array(process_landmark).reshape(1, 21, 2)
                time2 = time.time()
                prediction = model.predict(process_landmark_array, verbose=0)
                # print('only predict: {:2.2f} s'.format(time.time() - time2))
                prediction_index = np.argmax(prediction)
                # write_pre(prediction_index, 'vote/prediction.csv')
                print(frame_count, gestures[prediction_index])
                ten_y.append(prediction_index)
                if len(ten_y) == windows:
                    most_action = max(set(ten_y), key=ten_y.count)
                    action = ten_y.count(most_action)
                    # print(prediction_index)
                    if vote <= action / windows and most_action != 9:
                        print(Fore.LIGHTCYAN_EX + f"DO THE ACTION {gestures[most_action]}")
                        write_pre(most_action, 'vote/window.csv')
                        print(Style.RESET_ALL)
                    ten_y.pop(0)



def find_accuracy():
  filename = 'vote/label.csv'
  df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
  label = df.values.tolist()


  filename = 'vote/prediction.csv'
  df = pd.read_csv(filename, header=None)  # Read CSV file into a DataFrame without header
  prediction = df.values.tolist()

  print(len(prediction), len(label))


  trues = 0
  for i in range (len(prediction)):
    if prediction[i] == label[i]:
        trues += 1
  
  print('accuracy is ', trues/len(label))
    



capture_image()
# find_accuracy()
























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
# import urllib
# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import csv
# import cv2
# # from google.colab.patches import cv2_imshow



# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import math


# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision


# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np

# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


# # def pre_process_landmark(landmark_list):
# #     landmark_list = [[mp_normalized_landmark.x, mp_normalized_landmark.y,  mp_normalized_landmark.z] for mp_normalized_landmark in landmark_list]
# #     temp_landmark_list = copy.deepcopy(landmark_list)
# #     base_x, base_y = 0.0, 0.0 
# #     final_list = []
# #     for index in range(len(landmark_list)):
# #         if index == 0:
# #             base_x, base_y = landmark_list[index][0], landmark_list[index][1]

# #         x = temp_landmark_list[index][0] - base_x
# #         y = temp_landmark_list[index][1] - base_y
# #         final_list.append([x,y])

# #     return final_list



# def pre_process_landmark(landmark):
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
        

# def extract_coordinates(rgb_image, detection_result):
#     point_info = []
#     hand_landmarks_list = detection_result.hand_landmarks
#     # print('\n\n\n', 'hand_landmarks_list', hand_landmarks_list) ###### whole thing both hands
#     handedness_list = detection_result.handedness
#     annotated_image = np.copy(rgb_image)
#     for idx in range(len(hand_landmarks_list)):
#         hand_landmarks = hand_landmarks_list[idx]
#         # print('\n hand_landmarks', hand_landmarks)   ###### whole thing
#         handedness = handedness_list[idx]
#         # print('handedness', handedness)

        
#         # print('\n hand_landmarks', hand_landmarks)
#         hand_landmarks_flat = np.array(pre_process_landmark(hand_landmarks))
#         print('\n ', hand_landmarks_flat)
#         point_info.append(hand_landmarks_flat)
        
#         for category in handedness:
#             print(category.display_name)

#     return point_info


# def draw_landmarks_on_image(image, results):
#         mp_drawing = mp.solutions.drawing_utils
#         mp_hands = mp.solutions.hands
#         landmark_coords = []
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 for landmark in hand_landmarks.landmark:
#                     landmark_x = landmark.x * image.shape[1]
#                     landmark_y = landmark.y * image.shape[0]
#                     landmark_coords.append((landmark_x, landmark_y))
#         # print('landmark_coords' ,landmark_coords)
#         cv2.imwrite('./lol.jpg', image)
#         return image, landmark_coords
  
  

# def capture_image():
#     # Initialize MediaPipe Hand Landmarker
#     mp_hands = mp.solutions.hands
#     mp_drawing = mp.solutions.drawing_utils

#     # Initialize VideoCapture
#     cap = cv2.VideoCapture(0)
#     frame_count = 0
#     save_interval = 5  # seconds
#     output_folder = "./gestures/test/"
#     os.makedirs(output_folder, exist_ok=True)

#     # Initialize MediaPipe Hands Detector
#     options = {
#         "min_detection_confidence": 0.5,
#         "min_tracking_confidence": 0.5
#     }
#     detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


#     # Open CSV file
#     with open('./dataset/my_dataset_new.csv', 'a', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             cv2.imshow("Frame", frame)

#             if frame_count % save_interval == 0:
#                 # Convert the frame to RGB format
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 # Convert the RGB frame to MediaPipe's image format
#                 image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # BGR format for MediaPipe
#                 image.flags.writeable = False  # Set writable flag to False
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB

#                 # Detect hand landmarks in the image
#                 detection_result = detector.process(image)

#                 if detection_result.multi_hand_landmarks:
#                     # Draw landmarks on the image
#                     # print('detection_result', detection_result)
#                     annotated_image, landmark_coords = draw_landmarks_on_image(frame, detection_result)
#                     # print('landmark_coords', landmark_coords)

#                     # Save the annotated image
#                     cv2.imwrite(os.path.join(output_folder, f"annotated_frame_{frame_count}.jpg"), annotated_image)

#                     # Extract coordinates from detection result
#                     process_landmark = pre_process_landmark(np.array(landmark_coords))
#                     print(process_landmark, '\n\n')

#                     # Check for key press
#                     key = cv2.waitKey(1)
#                     if key != -1:
#                         if ord('0') <= key <= ord('9') and len(process_landmark) != 0:
#                             for point in process_landmark:
#                                 row_data = [chr(key)] + point[:42]
#                                 csvwriter.writerow(row_data)
#                                 print(f"Saved a list for {chr(key)}\n\n")
#                                 # break

#             # Exit when 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     # Release the camera and close the CSV file
#     cap.release()
#     cv2.destroyAllWindows()

# def main():
#     # STEP 2: Create an HandLandmarker object.
#     base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#     options = vision.HandLandmarkerOptions(base_options=base_options,
#                                         num_hands=2)
#     detector = vision.HandLandmarker.create_from_options(options)

#     # STEP 3: Load the input image.
#     # image = mp.Image.create_from_file("./gestures/annotated_frame21.jpg")
#     image = mp.Image.create_from_file("./gestures/2peace.jpg")
#     detection_result = detector.detect(image)
#     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#     print(detection_result)
#     cv2.imwrite('./gestures/lol.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


#     ##this foe model intry
#     annotated_image = extract_coordinates(image.numpy_view(), detection_result)


# # if __name__ == "__main__":
# #     # main()
# #     capture_image()



























# # #@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
# # from matplotlib import pyplot as plt
# # import mediapipe as mp
# # from mediapipe.framework.formats import landmark_pb2

# # plt.rcParams.update({
# #     'axes.spines.top': False,
# #     'axes.spines.right': False,
# #     'axes.spines.left': False,
# #     'axes.spines.bottom': False,
# #     'xtick.labelbottom': False,
# #     'xtick.bottom': False,
# #     'ytick.labelleft': False,
# #     'ytick.left': False,
# #     'xtick.labeltop': False,
# #     'xtick.top': False,
# #     'ytick.labelright': False,
# #     'ytick.right': False
# # })

# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # mp_drawing_styles = mp.solutions.drawing_styles


# # def display_one_image(image, title, subplot, titlesize=16):
# #     """Displays one image along with the predicted category name and score."""
# #     plt.subplot(*subplot)
# #     plt.imshow(image)
# #     if len(title) > 0:
# #         plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
# #     return (subplot[0], subplot[1], subplot[2]+1)


# # def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
# #     """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
# #     # Images and labels.
# #     images = [image.numpy_view() for image in images]
# #     gestures = [top_gesture for (top_gesture, _) in results]
# #     multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

# #     # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
# #     rows = int(math.sqrt(len(images)))
# #     cols = len(images) // rows

# #     # Size and spacing.
# #     FIGSIZE = 13.0
# #     SPACING = 0.1
# #     subplot=(rows,cols, 1)
# #     if rows < cols:
# #         plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
# #     else:
# #         plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

# #     # Display gestures and hand landmarks.
# #     for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
# #         title = f"{gestures.category_name} ({gestures.score:.2f})"
# #         dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
# #         annotated_image = image.copy()

# #         for hand_landmarks in multi_hand_landmarks_list[i]:
# #           hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
# #           hand_landmarks_proto.landmark.extend([
# #             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
# #           ])

# #           mp_drawing.draw_landmarks(
# #             annotated_image,
# #             hand_landmarks_proto,
# #             mp_hands.HAND_CONNECTIONS,
# #             mp_drawing_styles.get_default_hand_landmarks_style(),
# #             mp_drawing_styles.get_default_hand_connections_style())

# #         subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

# #     # Layout.
# #     plt.tight_layout()
# #     plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
# #     plt.show()




# # # def resize_and_show(image):
# # #   h, w = image.shape[:2]
# # #   if h < w:
# # #     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
# # #   else:
# # #     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# # # #   cv2_imshow(img)
# # #     # cv2.imshow(img)
# # #     print('hiiiiii')
# # #     cv2.imwrite('./gestures/ttt.jpg', img)
# # #     return img




# # IMAGE_FILENAMES = [ './gestures/thumbs_up.jpg','./gestures/2peace.jpg']
# # # IMAGE_FILENAMES = [ 'victory.jpg','test.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

# # # for name in IMAGE_FILENAMES:
# # #   url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
# # #   urllib.request.urlretrieve(url, name)

# # DESIRED_HEIGHT = 480
# # DESIRED_WIDTH = 480

# # # # Preview the images.
# # # images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# # # for name, image in images.items():
# # #   print(1, image.shape)
# # #   image = resize_and_show(image)
# # #   print(2, image.shape)



# # # STEP 1: Import the necessary modules.
# # import mediapipe as mp
# # from mediapipe.tasks import python
# # from mediapipe.tasks.python import vision

# # # STEP 2: Create an GestureRecognizer object.
# # base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
# # options = vision.GestureRecognizerOptions(base_options=base_options)
# # recognizer = vision.GestureRecognizer.create_from_options(options)

# # images = []
# # results = []
# # for image_file_name in IMAGE_FILENAMES:
# #   # STEP 3: Load the input image.
# #   print()
# #   image = mp.Image.create_from_file(image_file_name)

# #   # STEP 4: Recognize gestures in the input image.
# #   recognition_result = recognizer.recognize(image)

# #   # STEP 5: Process the result. In this case, visualize it.
# #   images.append(image)
# #   top_gesture = recognition_result.gestures[0][0]
# #   hand_landmarks = recognition_result.hand_landmarks
# #   results.append((top_gesture, hand_landmarks))

# #   print(image_file_name, hand_landmarks, '\n')

# # # display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
