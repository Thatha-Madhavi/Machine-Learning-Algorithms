import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, A, b):
        self.n_features = A.shape[1] if not self.n_features else min(A.shape[1],self.n_features)
        self.root = self._grow_tree(A, b)

    def _grow_tree(self, A, b, depth=0):
        n_samples, n_feats = A.shape
        n_labels = len(np.unique(b))

        # check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(b)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(A, b, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(A[:, best_feature], best_thresh)
        left = self._grow_tree(A[left_idxs, :], b[left_idxs], depth+1)
        right = self._grow_tree(A[right_idxs, :], b[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, A, b, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            A_column = A[:, feat_idx]
            thresholds = np.unique(A_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(b, A_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, b, A_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(b)

        # create children
        left_idxs, right_idxs = self._split(A_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # calculate the weighted avg. entropy of children
        n = len(b)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(b[left_idxs]), self._entropy(b[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, A_column, split_thresh):
        left_idxs = np.argwhere(A_column <= split_thresh).flatten()
        right_idxs = np.argwhere(A_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, b):
        hist = np.bincount(b)
        ps = hist / len(b)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, b):
        counter = Counter(b)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, A):
        return np.array([self._traverse_tree(a, self.root) for a in A])

    def _traverse_tree(self, a, node):
        if node.is_leaf_node():
            return node.value

        if a[node.feature] <= node.threshold:
            return self._traverse_tree(a, node.left)
        return self._traverse_tree(a, node.right)
        