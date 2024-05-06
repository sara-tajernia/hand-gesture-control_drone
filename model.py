import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv3D,MaxPooling3D, Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, LSTM,  ConvLSTM2D, Dropout, Input, Reshape
from keras.layers import TimeDistributed
import numpy as np
import time
import cv2
import tensorflow as tf
# from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from keras.models import Model
from keras.optimizers import Adam

class HandGestureClassifierMLP:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001

        self.input_shape = self.X_train.shape[1:3]  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()


    def build_model(self):
        # Define input layer
        input_layer = Input(shape=self.input_shape)

        # Flatten input
        flattened_input = Flatten()(input_layer)

        # Dense layers
        dense_1 = Dense(64, activation='relu')(flattened_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        dense_3 = Dense(16, activation='relu')(dense_2)

        # Output layer
        output_layer = Dense(self.actions_num, activation='sigmoid')(dense_3)

        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)

        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def train_model(self):
        start_train = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=200, batch_size=64, verbose=2)
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
        self.model.save('models/MLP(200).h5')

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
        self.learning_rate = 0.001

        self.input_shape = (self.X_train.shape[1:3])  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()


    def build_model(self):
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2, padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.actions_num, activation='softmax')
        ])

        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model


    def train_model(self):

        start_train = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=200, batch_size=64, verbose=2)
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
        self.model.save('models/CNN(200).h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)





class HandGestureClassifierLSTM:
    def __init__(self,X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001


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

        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
        self.model.save('models/LSTM(200).h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

    
































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





##### yolo

# def build_model(self):
#     model = Sequential([
#         Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, kernel_size=(3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(128, kernel_size=(3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(64, kernel_size=(3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Conv2D(32, kernel_size=(3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(self.actions_num, activation='softmax')
#     ])

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model


class HandGestureClassifierYolo:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001

        self.input_shape = self.X_train.shape[1:3]  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        # self.train_model()
        # self.save_model()
        # self.test_model()


    def build_model(self):
        # Define input layer
        # inputs = Input(shape=self.input_shape)
        print(self.input_shape)

        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=self.input_shape),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            Dense(128, activation='relu'),
            Dense(self.actions_num, activation='softmax')
        ])

        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def train_model(self):
        start_train = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=20, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy Yolo')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='test_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Yolo')
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save('models/Yolo(200).h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

    



class HandGestureClassifierVGG16:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001

        self.input_shape = self.X_train.shape[1:3]  #(21, 2)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()


    def build_model(self):
        model = Sequential()

        # Block 1
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Flatten
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.actions_num, activation='sigmoid'))

        # Compile model
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def train_model(self):
        start_train = time.time()
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=200, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='train_accuracy')
        plt.plot(self.history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy VGG-16')
        plt.legend()
        plt.show()

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='train_loss')
        plt.plot(self.history.history['val_loss'], label='test_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss VGG-16')
        plt.legend()
        plt.show()

    def save_model(self):
        self.model.save('models/VGG16(200).h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

    