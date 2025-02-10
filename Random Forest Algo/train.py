from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
A = data.data
b = data.target

A_train, A_test, b_train, b_test = train_test_split(
    A, b, test_size=0.2, random_state=1234
)

def accuracy(b_true, b_pred):
    accuracy = np.sum(b_true == b_pred) / len(b_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(A_train, b_train)
predictions = clf.predict(A_test)

acc =  accuracy(b_test, predictions)
print(acc)