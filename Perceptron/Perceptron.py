import numpy as np


def unit_step_func(a):
    return np.where(a > 0 , 1, 0)

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, A, b):
        n_samples, n_features = A.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        b_ = np.where(b > 0 , 1, 0)

        # learn weights
        for _ in range(self.n_iters):
            for idx, a_i in enumerate(A):
                linear_output = np.dot(a_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron update rule
                update = self.lr * (b_[idx] - b_predicted)
                self.weights += update * a_i
                self.bias += update


    def predict(self, A):
        linear_output = np.dot(A, self.weights) + self.bias
        b_predicted = self.activation_func(linear_output)
        return b_predicted


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(b_true, b_pred):
        accuracy = np.sum(b_true == b_pred) / len(b_true)
        return accuracy

    A, b = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    A_train, A_test, b_train, b_test = train_test_split(
        A, b, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(A_train, b_train)
    predictions = p.predict(A_test)

    print("Perceptron classification accuracy", accuracy(b_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(A_train[:, 0], A_train[:, 1], marker="o", c=b_train)

    a0_1 = np.amin(A_train[:, 0])
    a0_2 = np.amax(A_train[:, 0])

    a1_1 = (-p.weights[0] * a0_1 - p.bias) / p.weights[1]
    a1_2 = (-p.weights[0] * a0_2 - p.bias) / p.weights[1]

    ax.plot([a0_1, a0_2], [a1_1, a1_2], "k")

    ymin = np.amin(A_train[:, 1])
    ymax = np.amax(A_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()