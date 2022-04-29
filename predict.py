import numpy as np

def predict(self, X):
    X = self.add_ones(X.values)
    ypred = np.dot(X,self.theta)
    return ypred
    

def mse(self, y,yPred):
    mse =  np.sum((yPred - y) ** 2 ) / 2
    return mse