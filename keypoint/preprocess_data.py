import numpy as np
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self, X, y):
        self.X = []
        self.y = []
        self.coordination(X)
        self.extract_labels(y)
        
    def coordination(self, X):
        self.X = np.reshape(X, (X.shape[0], -1, 2))

    def extract_labels(self, y):
        labels = []
        for i in y:
            arr=[]
            for j in range(8):
                if(i == j):
                    arr.append(1)
                else:
                    arr.append(0)
            labels.append(arr)
        self.y = labels

        
    
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.99, random_state=42)   
        return X_train, X_test, y_train, y_test
