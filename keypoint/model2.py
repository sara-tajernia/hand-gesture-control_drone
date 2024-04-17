import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv3D,MaxPooling3D, Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, LSTM,  ConvLSTM2D, Dropout, Input
from keras.layers import TimeDistributed
import numpy as np
import time
import cv2
import tensorflow as tf
# from keras.models import Model
from keras.models import load_model



class HandGestureClassifierMLP:
    def __init__(self,X_train, y_train, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        self.input_shape = (self.X_train.shape[1:3])  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()


    def build_model(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),  # Flatten the input to make it compatible with Dense layers
            Dense(64, activation='relu'),
            # Dropout(0.0),
            Dense(32, activation='relu'),
            # Dropout(0.0),
            Dense(16, activation='relu'),
            Dense(self.actions_num, activation='sigmoid')  
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def train_model(self):
        start_train = time.time()
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))

    
    def save_model(self):
        self.model.save('models/weights_CNN_100.h5')






class HandGestureClassifierCNN:
    def __init__(self,X_train, y_train, actions_num):
        # self.X_train = np.array(X_train)
        # self.y_train = np.array(self.extract_labels(y_train))
        # self.X_train = np.reshape(X_train, (X_train.shape[0], -1, 2))

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        # self.test_model()


    def build_model(self):


        # self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1, 2))

        input_shape = (21, 2)

        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2, padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.actions_num, activation='softmax')
        ])


        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    


    def train_model(self):

        start_train = time.time()

        self.model.fit(self.X_train, self.y_train, epochs=500, batch_size=64, verbose=2)

        print('Training model: {:2.2f} s'.format(time.time() - start_train))

    
    def save_model(self):
        self.model.save('models/weights_CNN_100.h5')


    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))

        savedModel=load_model('models/weights_CNN.h5')
        test_loss, test_acc = savedModel.evaluate(self.X_test, self.y_test)
        print('LOAD Test accuracy: {:2.2f}%'.format(test_acc*100))







class  HandGestureClassifierCNNLSTM:
    # def __init__(self, X, y, actions_num):
    #     self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(X, y)
    #     self.actions_num = actions_num
    #     self.model = self.build_model()
    #     self.train_model()
    #     self.test_model()

    def __init__(self,X_train, y_train, actions_num):
        self.X_train = X_train = np.array(X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        self.y_train = y_train
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()


    def load_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = np.array(X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        X_test = np.array(X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
        # X_train = np.array(X_train.reshape(501, 1, 384,216,3))
        # X_test = np.array(X_test.reshape(126, 1, 384,216,3))

        return X_train, X_test, y_train, y_test

    def build_model(self):

        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (5,5), activation='relu'),input_shape=self.X_train.shape[1:5]))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.actions_num, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # print('hi', self.X_train.shape[1:3])
        model.summary()



        return model

    def train_model(self):
        start_train = time.time()
        self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))

    def save_model(self):
        self.model.save('models/weights_CNNLSTM.h5')


    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))




















