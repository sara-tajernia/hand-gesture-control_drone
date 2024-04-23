from keras.models import load_model
import numpy as np

class TestModel:
    def __init__(self,X_test, y_test):
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.test_model()


    def test_model(self):
        savedModel=load_model('models/weights_CNN_my1100.h5')
        test_loss, test_acc = savedModel.evaluate(self.X_test, self.y_test)
        print('Test accuracy: {:2.2f}%'.format(test_acc*100))
        print('Test loss: {:2.2f}%'.format(test_loss*100))





