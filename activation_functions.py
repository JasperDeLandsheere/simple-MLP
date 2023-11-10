import numpy as np

class ActivationFunction():
    """Base class for activation functions."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the activation function.

        Parameters:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Result of the forward pass.
        """
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass (derivative) of the activation function.

        Parameters:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Derivative of the activation function.
        """
        raise NotImplementedError("Backward pass not implemented.")

class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        sigma = 1 / (1 + np.exp(-x))
        return sigma * (1 - sigma)

class TanH(ActivationFunction):
    """Hyperbolic tangent (tanh) activation function."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0, x)      

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0, 1)

