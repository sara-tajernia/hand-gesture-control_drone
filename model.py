
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.metrics import classification_report, confusion_matrix 
from keras.layers import Dense, Flatten, LSTM, Dropout, Input, TimeDistributed

class HandGestureClassifierMLP:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001

        self.input_shape = self.X_train.shape[1:3] 
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
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=150, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Create a single plot for all metrics
        plt.figure(figsize=(10, 6))

        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(self.history.history['val_accuracy'], label='Test Accuracy', color='orange')

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='Train Loss', color='green')
        plt.plot(self.history.history['val_loss'], label='Test Loss', color='red')

        # Adding labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Test Accuracy & Loss MLP Model for Left Hand')
        plt.legend()

        # Show the plot
        plt.show()


    def save_model(self):
        self.model.save('models/MLP_Left(7150).h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

        cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        self.plot_confusion_matrix(cm, classes=[str(i) for i in range(self.actions_num)])
    

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix For MLP Model (Left Hand)', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    



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
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape, self.input_shape)
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=200, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Create a single plot for all metrics
        plt.figure(figsize=(10, 6))

        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(self.history.history['val_accuracy'], label='Test Accuracy', color='orange')

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='Train Loss', color='green')
        plt.plot(self.history.history['val_loss'], label='Test Loss', color='red')

        # Adding labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Test Accuracy & Loss CNN Model for Left Hand')
        plt.legend()

        # Show the plot
        plt.show()

    
    def save_model(self):
        self.model.save('models/me_left.h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

        cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        self.plot_confusion_matrix(cm, classes=[str(i) for i in range(self.actions_num)])

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix For CNN Model (Left Hand)', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()




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
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=150, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Create a single plot for all metrics
        plt.figure(figsize=(10, 6))

        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(self.history.history['val_accuracy'], label='Test Accuracy', color='orange')

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='Train Loss', color='green')
        plt.plot(self.history.history['val_loss'], label='Test Loss', color='red')

        # Adding labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Test Accuracy & Loss LSTM Model for Left Hand')
        plt.legend()

        # Show the plot
        plt.show()

    
    def save_model(self):
        self.model.save('models/LSTM_Left(7150).h5')
        # self.model.save('models/weights_CNN_100.h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

        cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        self.plot_confusion_matrix(cm, classes=[str(i) for i in range(self.actions_num)])
    

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix For LSTM Model (Left Hand)', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


    





class HandGestureClassifierRNN:
    def __init__(self, X_train, y_train, X_test, y_test, actions_num):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.learning_rate = 0.001

        self.input_shape = self.X_train.shape[1:3]  # (Timesteps, Features)
        self.actions_num = actions_num
        self.model = self.build_model()
        self.train_model()
        self.save_model()
        self.test_model()


    def build_model(self):

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=self.input_shape),
            TimeDistributed(Dense(self.actions_num, activation='sigmoid')),
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
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape, self.input_shape)
        self.history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=150, batch_size=64, verbose=2)
        print('Training model: {:2.2f} s'.format(time.time() - start_train))
        self.plot_training_progress()
    
    def plot_training_progress(self):
        # Create a single plot for all metrics
        plt.figure(figsize=(10, 6))

        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='Train Accuracy', color='blue')
        plt.plot(self.history.history['val_accuracy'], label='Test Accuracy', color='orange')

        # Plot training and validation loss
        plt.plot(self.history.history['loss'], label='Train Loss', color='green')
        plt.plot(self.history.history['val_loss'], label='Test Loss', color='red')

        # Adding labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Test Accuracy & Loss RNN Model for Left Hand')
        plt.legend()

        # Show the plot
        plt.show()

    def save_model(self):
        self.model.save('models/RNN_Left(7150).h5')

    def test_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))
        predictions = self.model.predict(self.X_test)
        y_pred = (predictions > 0.5).astype(int)
        report = classification_report(self.y_test, y_pred)
        print(report)

        cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        self.plot_confusion_matrix(cm, classes=[str(i) for i in range(self.actions_num)])
    

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix For RNN Model (Left Hand)', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 7))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

