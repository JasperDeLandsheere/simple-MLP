import numpy as np

class LossFunction:
    """Base class for loss functions."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute the loss between the true and predicted values.

        Parameters:
        - y_true (np.ndarray): True values.
        - y_pred (np.ndarray): Predicted values.

        Returns:
        - float: Computed loss.
        """
        raise NotImplementedError("Loss computation not implemented.")

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predicted values.

        Parameters:
        - y_true (np.ndarray): True values.
        - y_pred (np.ndarray): Predicted values.

        Returns:
        - np.ndarray: Gradient of the loss.
        """
        raise NotImplementedError("Loss gradient computation not implemented.")

class MSE(LossFunction):
    """Mean Squared Error (MSE) loss function."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_pred.shape[0]
