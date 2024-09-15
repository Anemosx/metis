import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SupportVectorMachine:
    """
    Support Vector Machine (SVM) classifier using a linear kernel.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for weight updates (default is 0.001).
    lambda_param : float, optional
        Regularization parameter (default is 0.01).
    n_iters : int, optional
        Number of iterations for training (default is 1000).

    Attributes
    ----------
    w : np.ndarray
        Weights of the SVM model.
    b : float
        Bias term of the SVM model.

    Methods
    -------
    fit(X, y)
        Trains the SVM model on the given dataset.
    predict(X)
        Predicts the class labels for the input samples.
    get_hyperplane_value(x, offset)
        Calculates the value of the hyperplane for a given x coordinate and offset.
    get_hyperplane_formula()
        Returns the formula of the separating hyperplane.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_iters: int = 1000,
    ) -> None:
        """
        Initializes the SVM with specified learning rate, regularization parameter and number of iterations.
        """

        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the SVM model on the given dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target vector of shape (n_samples,), where each element is either 1 or -1.
        """

        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # update weights for samples satisfying the margin condition
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # update weights and bias for samples not satisfying the margin condition
                    self.w -= self.learning_rate * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for the input samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """

        approx = np.dot(X, self.w) - self.b

        return np.sign(approx)

    def get_hyperplane_value(self, x: float, offset: float) -> float:
        """
        Calculates the value of the hyperplane for a given x coordinate and offset.

        Parameters
        ----------
        x : float
            x-coordinate for which to calculate the hyperplane value.
        offset : float
            Offset to adjust the hyperplane (e.g., for margin boundaries).

        Returns
        -------
        float
            The y-coordinate of the hyperplane.
        """

        return (-self.w[0] * x + self.b + offset) / self.w[1]

    def get_hyperplane_formula(self) -> str:
        """
        Returns the formula of the separating hyperplane.

        Returns
        -------
        str
            The formula of the hyperplane as a string.
        """

        return f"{self.w[0]:.2f} * x1 + {self.w[1]:.2f} * x2 = {self.b:.2f}"


def plot_decision_boundary(
    X: np.ndarray, y: np.ndarray, model: SupportVectorMachine
) -> None:
    """
    Plots the decision boundary and margins for the SVM model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        True labels of the data of shape (n_samples,).
    model : SupportVectorMachine
        The trained SVM model.
    """

    flatui = ["#9b59b6", "#3498db"]
    fig, ax = plt.subplots()

    # calculate hyperplane boundaries
    x0_1, x0_2 = np.amin(X[:, 0]), np.amax(X[:, 0])
    x1_1 = model.get_hyperplane_value(x0_1, 0)
    x1_2 = model.get_hyperplane_value(x0_2, 0)
    x1_1_m = model.get_hyperplane_value(x0_1, -1)
    x1_2_m = model.get_hyperplane_value(x0_2, -1)
    x1_1_p = model.get_hyperplane_value(x0_1, 1)
    x1_2_p = model.get_hyperplane_value(x0_2, 1)

    # plot hyperplane and margins
    sns.lineplot(
        x=[x0_1, x0_2],
        y=[x1_1, x1_2],
        ax=ax,
        color="black",
        label=model.get_hyperplane_formula(),
    )
    ax.fill_between(
        [x0_1, x0_2], [x1_1_m, x1_2_m], [x1_1_p, x1_2_p], color="black", alpha=0.1
    )
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=flatui, edgecolor=None, ax=ax)

    ax.set_ylim([np.amin(X[:, 1]) - 3, np.amax(X[:, 1]) + 3])
    ax.legend()
    plt.show()


def run() -> None:
    """
    Runs the full pipeline for generating data, training the SVM and plotting the decision boundary.
    """

    # generate dataset
    dataset, labels = make_blobs(
        n_samples=2000, n_features=2, centers=[(-1, -1), (1, 1)], cluster_std=0.5
    )

    # convert labels to -1 and 1
    labels = np.where(labels <= 0, -1, 1)
    train_set, test_set, train_labels, test_labels = train_test_split(
        dataset, labels, test_size=0.4
    )

    # normalize features
    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)

    # train the SVM
    svm = SupportVectorMachine()
    svm.fit(train_set, train_labels)

    # predict and evaluate
    predictions = svm.predict(test_set)
    print(f"Separating hyperplane: {svm.get_hyperplane_formula()}")
    accuracy = np.mean(predictions == test_labels)
    print(f"SVM classification accuracy: {accuracy:.2f}")

    # plot decision boundary
    plot_decision_boundary(test_set, test_labels, svm)


if __name__ == "__main__":
    run()
