import cv2
import mediapipe as mp
import time
# from google.colab.patches import cv2_imshow

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2



MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image










img = cv2.imread("./gestures/Unknown.png")
# cv2.imshow('', img)
cv2.imwrite('./gestures/hi.png', img)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("./gestures/Unknown.png")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)



# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
images = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('./gestures/hi2.png', images)
# cv2.imwrite(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

























# class HandDetector():
#     def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCon = 0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.modelComplex = modelComplexity
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpHands = mp.solutions.hands
#         self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon)
#         self.mpDraw = mp.solutions.drawing_utils

#     def findHands(self,img,draw=True):
#         img = cv2.flip(img,1)
#         imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         self.results = self.hands.process(imgRGB)
#         #print(results.multi_hand_landmarks)

#         if self.results.multi_hand_landmarks:
#             for handlms in self.results.multi_hand_landmarks:   
#                 '''
                
#                     '''
#                 if draw:
#                     self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)

#         return img
    
#     def findPosition(self,img,handNo=0,draw=True):
        
#         lmlist = []
#         if self.results.multi_hand_landmarks:
#             myHand=self.results.multi_hand_landmarks[handNo]

#             for id,lm in enumerate(myHand.landmark):
#                 h,w,c = img.shape
#                 cx,cy = int(lm.x*w),int(lm.y*h)
#                 #print(id,cx,cy)
#                 lmlist.append([id,cx,cy])
#                 if draw:
#                     cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

#         return lmlist
    

# mp_hands = mp.solutions.hands.Hands()
# frame = cv2.imread("./gestures/fist.jpeg")
# results = mp_hands.process(frame)

# if results.multi_hand_landmarks:
#     print('Hands were detected')
    
# else:
#     print('No hands were detected')