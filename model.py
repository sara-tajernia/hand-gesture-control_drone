import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split # Helps with organizing data for training
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten

class HandGestureClassifier:
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


    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)

        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        # predictions = model.predict(X_test)
        print(len(self.X_test))


  
