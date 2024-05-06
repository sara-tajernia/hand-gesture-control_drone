import numpy as np
from model import HandGestureClassifierMLP, HandGestureClassifierCNN, HandGestureClassifierYolo, HandGestureClassifierLSTM
import time
from test_model import TestModel
from preprocess_data import Preprocess
from handDetector import HandDetector

from dataset import Dataset




if __name__ == "__main__":

    # # Train
    # actions_num = 10
    # dataset = "./dataset/dataset_1hand(10).csv"
    # # dataset = "./dataset/keypoint.csv"
    # X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    # y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    # preprocess = Preprocess(X_dataset, y_dataset, actions_num)
    
    # X_train, X_test, y_train, y_test = preprocess.split_data()




    # model = HandGestureClassifierMLP(X_train, y_train, X_test, y_test, actions_num).model
    # model = HandGestureClassifierCNN(X_train, y_train, X_test, y_test, actions_num).model
    # model = HandGestureClassifierLSTM(X_train, y_train, X_test, y_test, actions_num).model
    # model = HandGestureClassifierYolo(X_train, y_train, X_test, y_test, actions_num).model



    # Uncomment if you want het orders for drone
    hand_detector = HandDetector()


    #Uncomment if if you want to see the accuracy of 20% of data
    # TestModel(X_test, y_test, 'models/MLP_1hand(10).h5')


    #Uncomment if you want to collect data to add to dataset
    # Dataset()


