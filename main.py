
import cv2
from preprocess import preprocess
import pandas as pd
import train
import numpy as np


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# from dataset import CustomDataset
from model import HandGestureClassifier
# from train import Trainer

# up = 0 // down = 1 // back = 2 // stop = 3 // land = 4 // front = 5 // right = 6 // left = 7  // none = 8     video1

# up = 0 // down = 1 // back = 2 // forward = 3 // land = 4 // stop = 5 // left = 6 // right = 7  // none = 8     video2



if __name__ == "__main__":
    # cap = cv2.VideoCapture('./input/video.mp4')
    cap = cv2.VideoCapture('./input/video2.mp4')

    pre_data = preprocess(cap=cap)
    X = pre_data.frames
    y = pre_data.labels
    print(len(X))
    print(len(y))
    model = HandGestureClassifier(X, y).model
    # print(model)

