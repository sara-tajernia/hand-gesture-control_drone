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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class HandGestureClassifierMLP:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        self.input_shape = self.X_train.shape[1:3]  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()


    def build_model(self):
        model = Sequential([
            Flatten(input_shape=self.input_shape),  # Flatten the input to make it compatible with Dense layers
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.actions_num, activation='sigmoid')  
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def train_model(self):
        start_train = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=50, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy MLP')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='test_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss MLP')
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save('models/MLP_1hand(10).h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

    



class HandGestureClassifierCNN:
    def __init__(self,X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        self.input_shape = (self.X_train.shape[1:3])  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        # self.save_model()
        self.test_model()


    def build_model(self):

        # input_shape = (21, 2)

        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=self.input_shape),
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
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy CNN')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='test_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss CNN')
        plt.legend()
        plt.show()

    
    def save_model(self):
        self.model.save('models/CNN_1hand(10).h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))





class HandGestureClassifierLSTM:
    def __init__(self,X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

        self.input_shape = (self.X_train.shape[1:3])  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()



    def build_model(self):

        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=self.input_shape),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(128, activation='relu'),
            Dense(self.actions_num, activation='softmax')
        ])


        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    
    def train_model(self):

        start_train = time.time()
        # self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=64, verbose=2)
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=200, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy LSTM')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='test_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss LSTM')
        plt.legend()
        plt.show()

    
    def save_model(self):
        self.model.save('models/LSTM_1hand(10).h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))

    






















    
        


class  HandGestureClassifierCNNLSTM:
    def __init__(self,X_train, y_train, actions_num):
        self.X_train = np.array(X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
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


















########    VGG-16


# def build_model(self):

#     model = Sequential()

#     # Block 1
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Block 2
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Block 3
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Block 4
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Block 5
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#     # Flatten
#     model.add(Flatten())

#     # Fully connected layers
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(self.actions_num, activation='sigmoid'))

#     # Compile the model
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Print model summary
#     model.summary()



