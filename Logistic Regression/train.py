import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()
A, b = bc.data, bc.target
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(A_train,b_train)
b_pred = clf.predict(A_test)

def accuracy(b_pred, b_test):
    return np.sum(b_pred==b_test)/len(b_test)

acc = accuracy(b_pred, b_test)
print(acc)