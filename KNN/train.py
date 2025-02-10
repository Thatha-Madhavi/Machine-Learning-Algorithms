import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
A, b = iris.data, iris.target

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(A[:,2],A[:,3], c=b, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNN(k=5)
clf.fit(A_train, b_train)
predictions = clf.predict(A_test)

print(predictions)

acc = np.sum(predictions == b_test) / len(b_test)
print(acc)