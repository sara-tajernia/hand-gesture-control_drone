import numpy as np
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self, X, y, actions_num):
        self.X = []
        self.y = []
        self.actions_num = actions_num
        self.coordination(X)
        self.extract_labels(y)
        # self.data_info()
        
    def coordination(self, X):
        self.X = np.reshape(X, (X.shape[0], -1, 2))


    def data_info(sel, y):
        ddd = {}
        for item in y:   # touches each item exactly once
            if item in ddd:       # but uses additional checks and conditions 
                ddd[item] += 1
            else:
                ddd[item] = 1

        print(ddd) 


    def extract_labels(self, y):
        # self.data_info(y)
        labels = []
        for i in y:
            arr=[]
            for j in range(self.actions_num):
                if(i == j):
                    arr.append(1)
                else:
                    arr.append(0)
            labels.append(arr)
        self.y = labels


    
    
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.99, random_state=42)   
        return X_train, X_test, y_train, y_test
