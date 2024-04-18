import cv2
import pandas as pd
# import train
import numpy as np
import os, shutil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# from dataset import CustomDataset
from model2 import HandGestureClassifierMLP, HandGestureClassifierCNN, HandGestureClassifierCNNLSTM
import time
from sklearn.model_selection import train_test_split
from test import Test
from test2 import Test2
from preprocess_data import Preprocess
from handDetector import HandDetector


if __name__ == "__main__":

    # # Train
    # actions_num = 8
    # dataset = "./dataset/keypoint.csv"
    # X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    # y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    # preprocess = Preprocess(X_dataset, y_dataset)
    
    # X_train, X_test, y_train, y_test = preprocess.split_data()


    # # model = HandGestureClassifierMLP(X_train, y_train, actions_num).model
    # model = HandGestureClassifierCNN(X_train, y_train, actions_num).model
    # # model = HandGestureClassifierCNNLSTM(X_train, y_train, actions_num).model



    # hand_detector = HandDetector()



    Test2()






    # #Test Window
    # start_reading_test = time.time()
    # X_test = np.array(data_test.frames)
    # X_test = np.array(X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
    # y_test = np.array(data_test.labels)

    # print('reading data: {:2.2f} s'.format(time.time() - start_reading_test))

    # print('test shape', X_test.shape)
    # print('test y shape', y_test.shape)

    # TestWindow(X_test, y_test)
    



    


