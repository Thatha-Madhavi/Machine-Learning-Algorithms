import numpy as np


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, A):
        # mean centering
        self.mean = np.mean(A, axis=0)
        A = A -  self.mean

        # covariance, functions needs samples as columns
        cov = np.cov(A.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # eigenvectors v = [:, i] column vector, transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, A):
        # projects data
        A = A - self.mean
        return np.dot(A, self.components.T)


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # data = datasets.load_digits()
    data = datasets.load_iris()
    A = data.data
    b = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(A)
    A_projected = pca.transform(A)

    print("Shape of A:", A.shape)
    print("Shape of transformed A:", A_projected.shape)

    a1 = A_projected[:, 0]
    a2 = A_projected[:, 1]

    plt.scatter(
        a1, a2, c=b, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.alabel("Principal Component 1")
    plt.blabel("Principal Component 2")
    plt.colorbar()
    plt.show()