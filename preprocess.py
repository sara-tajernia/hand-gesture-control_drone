import cv2
import pandas as pd
import numpy as np
import os
from os import listdir

class preprocess:
    def __init__(self, cap):
        self.cap = cap
        self.frames = []
        self.labels = []
        self.extract_frames()
        # self.read_frames()
        self.extract_labels()


    def read_frames(self):
        frame_num = 0
        while frame_num < 2737:
            frame_filename = f"output/frame_{frame_num}.png"
            img = cv2.imread(frame_filename)
            self.frames.append(img)


    def extract_frames(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return

        # Get video properties
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Total frames: {frame_count}")
        print(f"Frames per second: {fps}")

        # Calculate the frame interval to extract one frame per second
        # frame_interval = int(round(fps))
        frame_interval = 1

        # Read and save frames at one frame per second
        frame_number = 0
        while True:
            # Set the video capture object to the next second
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (216, 384))  #video1
                # frame = cv2.resize(frame, (256, 144))   #video2
                self.frames.append(frame)
                frame_filename = f"output/frame_{frame_number}.png"
                cv2.imwrite(frame_filename, frame)
            else:
                print(f"Error reading frame {frame_number}")
                break

            frame_number += frame_interval

            if frame_number >= frame_count:
                break

        # Release the capture object
        self.frames = np.array(self.frames)
        self.cap.release()

    def extract_labels(self):
        # df = pd.read_csv('./input/labels_9_classes.csv', header=None)
        # df = pd.read_csv('./input/labels_8_classes.csv', header=None)
        # df = pd.read_csv('./input/video2_labels.csv', header=None)
        df = pd.read_csv('./input/video_labels.csv', header=None)

        # self.labels = df.to_numpy()
        temp = df.values.flatten()
        label =[]
        for i in temp:
            arr=[]
            for j in range(9):
                if(i == j):
                    arr.append(1)
                else:
                    arr.append(0)
            label.append(arr)
        self.labels = label
 
