import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, A, b):
        n_samples, n_features = A.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(A, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(A.T, (predictions - b))
            db = (1/n_samples) * np.sum(predictions-b)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, A):
        linear_pred = np.dot(A, self.weights) + self.bias
        b_pred = sigmoid(linear_pred)
        class_pred = [0 if b<=0.5 else 1 for b in b_pred]
        return class_pred