import numpy as np
from collect_dataset import Dataset
from preprocess_data import Preprocess
from handDetector2 import HandDetector
from model import HandGestureClassifierMLP, HandGestureClassifierCNN, HandGestureClassifierLSTM, HandGestureClassifierRNN




def train_model(path_dataset):
    # Train
    actions_num = 10
    dataset = path_dataset
    X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    preprocess = Preprocess(X_dataset, y_dataset, actions_num)
    
    X_train, X_test, y_train, y_test = preprocess.split_data()

    HandGestureClassifierMLP(X_train, y_train, X_test, y_test, actions_num).model
    HandGestureClassifierCNN(X_train, y_train, X_test, y_test, actions_num).model
    HandGestureClassifierLSTM(X_train, y_train, X_test, y_test, actions_num).model
    HandGestureClassifierRNN(X_train, y_train, X_test, y_test, actions_num).model

def test_project():
    HandDetector('Left', 'models/me_right.h5')

def collect_dataset():
    Dataset('./dataset/left_hand.csv')



if __name__ == "__main__":

    collect_dataset()

    # path_dataset = "./dataset/me_left.csv"
    # train_model(path_dataset)


    # hand_id = 'Right'
    # if hand_id == 'Right':
    #     hand_detector = HandDetector('Left', 'models/me_right.h5')
    # elif hand_id == 'Left':
    #     hand_detector = HandDetector('Right', 'models/me_left.h5')

