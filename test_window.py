
from keras.models import load_model
import operator
from collections import Counter
import numpy as np
class TestWindow:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.windows = 10
        self.vote = 0.7
        self.test_model()


    def test_model(self):
        savedModel=load_model('model/weights.h5')
        y_pred_final = []
        # test_Xs = self.X_test[100:120]
        ten_y = []

        for X in self.X_test:
            # X_t = np.array(x.reshape(1, x.shape[0], x.shape[1], x.shape[2]))
            y_t = savedModel.predict(X)
            ten_y.append(y_t)
            y_preds_windows = []
            if len(ten_y) == self.windows:
                for window in ten_y:
                    index = np.argmax(window)
                    y_preds_windows.append(index)
                _, freq = Counter(y_preds_windows).most_common(1)[0]
                if len(y_preds_windows)*self.vote <= freq:
                    y_pred_final.append(max(enumerate(y_preds_windows))[1])
                    # y_pred_final.append(max(enumerate(y_preds_windows), key=operator.itemgetter(1))[1])
                else:
                    y_pred_final.append(10)
                ten_y.pop(0)
                
        # print('\n hiii \n', self.y_test[100:120])
        print('123', y_pred_final)

        for i in range(len(y_pred_final)):
            print(i-10, ': ', y_pred_final[i])



        # print(savedModel.predict(self.X_test[100:120]))

