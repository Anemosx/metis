import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SupportVectorMachine:
    """
    Support Vector Machine (SVM) classifier.

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate for gradient descent. Default is 0.001.
    lambda_param : float, optional
        The regularization parameter. Default is 0.01.
    n_iters : int, optional
        The number of iterations for training. Default is 1000.

    Attributes
    ----------
    w : np.ndarray, None
        The weights of the model after training.
    b : float, None
        The bias term of the model after training.
    """

    def __init__(self, learning_rate: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w: np.ndarray | None = None
        self.b: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on samples in X.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels for samples in X.
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def get_hyperplane_value(self, x: float, offset: float) -> float:
        """
        Calculate the value of the hyperplane at a given x value.

        Parameters
        ----------
        x : float
            The x value at which to calculate the hyperplane value.
        offset : float
            The offset for the hyperplane (e.g., 0 for the main hyperplane, +/-1 for the margins).

        Returns
        -------
        float
            The value of the hyperplane at the given x.
        """
        return (-self.w[0] * x + self.b + offset) / self.w[1]

    def get_hyperplane_formula(self) -> str:
        """
        Get the formula of the separating hyperplane.

        Returns
        -------
        str
            The formula of the hyperplane as a string.
        """
        return f"{self.w[0]:.2f} * x1 + {self.w[1]:.2f} * x2 = {self.b:.2f}"


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: SupportVectorMachine) -> None:
    """
    Plot the decision boundary of the SVM along with the data points.

    Parameters
    ----------
    X : np.ndarray
        Data points of shape (n_samples, n_features).
    y : np.ndarray
        Class labels of shape (n_samples,).
    model : SupportVectorMachine
        Trained SVM model.
    """
    flatui = ["#9b59b6", "#3498db"]
    fig, ax = plt.subplots()

    x0_1, x0_2 = np.amin(X[:, 0]), np.amax(X[:, 0])
    x1_1 = model.get_hyperplane_value(x0_1, 0)
    x1_2 = model.get_hyperplane_value(x0_2, 0)
    x1_1_m = model.get_hyperplane_value(x0_1, -1)
    x1_2_m = model.get_hyperplane_value(x0_2, -1)
    x1_1_p = model.get_hyperplane_value(x0_1, 1)
    x1_2_p = model.get_hyperplane_value(x0_2, 1)

    sns.lineplot(x=[x0_1, x0_2], y=[x1_1, x1_2], ax=ax, color='black', label=model.get_hyperplane_formula())
    ax.fill_between([x0_1, x0_2], [x1_1_m, x1_2_m], [x1_1_p, x1_2_p], color='black', alpha=0.1)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=flatui, edgecolor=None, ax=ax)

    ax.set_ylim([np.amin(X[:, 1]) - 3, np.amax(X[:, 1]) + 3])
    ax.legend()
    plt.show()


def run() -> None:
    """
    Run the SVM example with synthetic data, train the model, and plot the decision boundary.
    """
    dataset, labels = make_blobs(n_samples=2000, n_features=2, centers=[(-1, -1), (1, 1)], cluster_std=0.5)
    labels = np.where(labels <= 0, -1, 1)
    train_set, test_set, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.4)

    scaler = StandardScaler()
    train_set = scaler.fit_transform(train_set)
    test_set = scaler.transform(test_set)

    svm = SupportVectorMachine()
    svm.fit(train_set, train_labels)
    predictions = svm.predict(test_set)

    print(f'Separating hyperplane: {svm.get_hyperplane_formula()}')
    accuracy = np.mean(predictions == test_labels)
    print(f'SVM classification accuracy: {accuracy:.2f}')

    plot_decision_boundary(test_set, test_labels, svm)


if __name__ == "__main__":
    run()
