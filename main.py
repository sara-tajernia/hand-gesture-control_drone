import numpy as np
from model import HandGestureClassifierMLP, HandGestureClassifierCNN, HandGestureClassifierCNNLSTM
import time
from test_model import TestModel
from preprocess_data import Preprocess
from handDetector import HandDetector

# from dataset import Dataset




if __name__ == "__main__":

    # # Train
    # actions_num = 9
    # dataset = "./dataset/my_dataset_new.csv"
    # # dataset = "./dataset/keypoint.csv"
    # X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
    # y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

    # preprocess = Preprocess(X_dataset, y_dataset, actions_num)
    
    # X_train, X_test, y_train, y_test = preprocess.split_data()
    # # print(X_train[0])




    # # model = HandGestureClassifierMLP(X_train, y_train, actions_num).model
    # model = HandGestureClassifierCNN(X_train, y_train, actions_num).model
    # # model = HandGestureClassifierCNNLSTM(X_train, y_train, actions_num).model



    # Uncomment if you want het orders for drone
    hand_detector = HandDetector()


    #Uncomment if if you want to see the accuracy of 20% of data
    # TestModel(X_test, y_test)


    #Uncomment if you want to collect data to add to dataset
    # Dataset()






    # #Test Window
    # start_reading_test = time.time()
    # X_test = np.array(data_test.frames)
    # X_test = np.array(X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
    # y_test = np.array(data_test.labels)

    # print('reading data: {:2.2f} s'.format(time.time() - start_reading_test))

    # print('test shape', X_test.shape)
    # print('test y shape', y_test.shape)

    # TestWindow(X_test, y_test)
    



    


