import cv2
import pandas as pd
import numpy as np


class preprocess:
    def __init__(self, cap):
        self.cap = cap
        self.frames = []
        self.labels = []
        self.extract_frames()
        self.extract_labels()

    def extract_frames(self):
        # frames = []
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Total frames: {frame_count}")
        print(f"Frames per second: {fps}")

        # Calculate the frame interval to extract one frame per second
        frame_interval = int(round(fps))

        # Read and save frames at one frame per second
        frame_number = 0
        while True:
            # Set the video capture object to the next second
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = self.cap.read()
            # print(rat)

            # If the frame was read successfully, save it
            if ret:
                frame = cv2.resize(frame, (216, 384))
                # frame = cv2.resize(frame, (320, 120))
                self.frames.append(frame)
                frame_filename = f"output/frame_{frame_number}.png"
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"Error reading frame {frame_number}")
                break

            # Move to the next second
            frame_number += frame_interval

            # Break the loop if we reach the end of the video
            if frame_number >= frame_count:
                break

        # Release the capture object
        self.frames = np.array(self.frames)
        self.cap.release()

    def extract_labels(self):
        df = pd.read_csv('./input/labels.csv', header=None)
        self.labels = df.values.flatten()
 

# class

# def extract_frames(cap):
#     frames = []
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     # Get video properties
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     print(f"Total frames: {frame_count}")
#     print(f"Frames per second: {fps}")

#     # Calculate the frame interval to extract one frame per second
#     frame_interval = int(round(fps))

#     # Read and save frames at one frame per second  

#     frame_number = 0
#     while True:
#         # Set the video capture object to the next second
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

#         # Read the frame
#         ret, frame = cap.read()

#         # If the frame was read successfully, save it
#         if ret:
#             frames.append(frame)
#             frame_filename = f"output/frame_{frame_number}.png"
#             cv2.imwrite(frame_filename, frame)
#         else:
#             print(f"Error reading frame {frame_number}")
#             break

#         # Move to the next second
#         frame_number += frame_interval

#         # Break the loop if we reach the end of the video
#         if frame_number >= frame_count:
#             break

#     # Release the capture object
#     cap.release()




