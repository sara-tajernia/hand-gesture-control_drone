import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv3D,MaxPooling3D
from keras.layers import Dense, Flatten, LSTM
from keras.layers import TimeDistributed
import numpy as np
class HandGestureClassifierCNN:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(X, y)
        self.model = self.build_model()
        self.train_model()
        self.test_model()

    def load_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        model = Sequential([
            # Conv2D(32, (5, 5), activation='relu', input_shape=(144, 256, 3)),
            Conv2D(32, (5, 5), activation='relu', input_shape=(384, 216, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(9, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=64, verbose=2)
        # self.model.fit(np.array(self.X_train), np.array(self.y_train), epochs=5, batch_size=64, verbose=2)


    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))






class HandGestureClassifierCNNLSTM:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data(X, y)
        self.model = self.build_model()
        self.train_model()
        # self.test_model()

    def load_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train)
        return X_train, X_test, y_train, y_test

    def build_model(self):
        # model = Sequential([
        #     Conv2D(32, (5, 5), activation='relu', input_shape=(144, 256, 3)),
        #     # Conv2D(32, (5, 5), activation='relu', input_shape=(384, 216, 3)),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     MaxPooling2D((2, 2)),
        #     Conv2D(64, (3, 3), activation='relu'),
        #     MaxPooling2D((2, 2)),
        #     LSTM(64, return_sequences=True, activation='relu'),
        #     LSTM(128, return_sequences=True, activation='relu'),
        #     LSTM(64, return_sequences=False, activation='relu'),
        #     Flatten(),
        # ])

        # # model.add(LSTM(64, return_sequences=True, activation='relu'))
        # # model.add(LSTM(128, return_sequences=True, activation='relu'))
        # # model.add(LSTM(64, return_sequences=False, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(9, activation='softmax'))


        cnn = Sequential()
        # define CNN model
        cnn.add(Conv2D(32, (5, 5), activation='relu', input_shape=(384, 216, 3)))
        # cnn.add(MaxPooling2D((2, 2)))
        cnn.add(Flatten())
        model = Sequential()
        model.add(TimeDistributed(cnn))
        model.add(LSTM(32, return_sequences=True, activation='relu'))
        model.add(Dense(9, activation='relu'))


        # print('hiiiii',model.compute_output_shape((None,384, 216, 3)))
        # print('hiii2', self.X_train.shape)
        # print('hiii2', self.y_train.shape)





        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self):
        # print(len(self.X_train[0]))
        self.model.fit(np.array([self.X_train]), np.array([self.y_train]), epochs=5, batch_size=16, verbose=2)


    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)

        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        # predictions = model.predict(X_test)
        print(len(self.X_test))


    