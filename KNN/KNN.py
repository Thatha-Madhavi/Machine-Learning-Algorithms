# We need to calculate it's distance from all other data points.
# Get closest k-points.
# In Regression - We will get the average of their values.
# In Classification - Will get the label with majority vote.

import numpy as np
from collections import Counter

def euclidean_distance(a1, a2):
    distance = np.sqrt(np.sum((a1-a2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, a, b):
        self.A_train = a
        self.b_train = b

    def predict(self, A):
        predictions = [self._predict(a) for a in A]
        return predictions

    def _predict(self, a):
        # compute the distance
        distances = [euclidean_distance(a, a_train) for a_train in self.A_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.b_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]