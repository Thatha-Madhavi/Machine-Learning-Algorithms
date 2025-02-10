from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTree

data = datasets.load_breast_cancer()
A, b = data.data, data.target

A_train, A_test, b_train, b_test = train_test_split(
    A, b, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(A_train, b_train)
predictions = clf.predict(A_test)

def accuracy(b_test, b_pred):
    return np.sum(b_test == b_pred) / len(b_test)

acc = accuracy(b_test, predictions)
print(acc)