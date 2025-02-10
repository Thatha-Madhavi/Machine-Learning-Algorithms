import numpy as np

class NaiveBayes:

    def fit(self, A, b):
        n_samples, n_features = A.shape
        self._classes = np.unique(b)
        n_classes = len(self._classes)

        # calculate mean, var, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            A_c = A[y == c]
            self._mean[idx, :] = A_c.mean(axis=0)
            self._var[idx, :] = A_c.var(axis=0)
            self._priors[idx] = A_c.shape[0] / float(n_samples)
            

    def predict(self, A):
        b_pred = [self._predict(a) for x in A]
        return np.array(b_pred)

    def _predict(self, a):
        posteriors = []

        # calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, a)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, a):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((a - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(b_true, b_pred):
        accuracy = np.sum(b_true == b_pred) / len(b_true)
        return accuracy

    A, b = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    A_train, A_test, b_train, b_test = train_test_split(
        A, b, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(A_train, b_train)
    predictions = nb.predict(A_test)

    print("Naive Bayes classification accuracy", accuracy(b_test, predictions))