import numpy as np

class MLP:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activation_func):
        """
        Parameters
        ----------
        input_size: int
            Size of the input vector.
        hidden_size: int
            Size of the hidden layer.
        output_size: int
            Size of the output vector.
        activation_func: ActivationFunction
            Activation function of your choice.
        """
        self.W_h = np.random.normal(0, 1, (input_size, hidden_size))
        self.b_h = np.random.normal(0, 1, (1, hidden_size))
        self.W_o = np.random.normal(0, 1, (hidden_size, output_size))
        self.b_o = np.random.normal(0, 1, (1, output_size))
        self.activation_func = activation_func
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MLP.
        
        Parameters
        ----------
        x: np.ndarray
            Input vector of size (input_size).
            
        Returns
        -------
        y: np.ndarray
            Output vector of size (output_size).
        """
        y, _, _ = self.forward_(x)
        return y
    
    def forward_(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass of the MLP with additional return values.
        
        Parameters
        ----------
        x: np.ndarray
            Input vector of size (input_size).
            
        Returns
        -------
        y: np.ndarray
            Output vector of size (output_size).
        h: np.ndarray
            Activation of the hidden layer of size (hidden_size).
        z_h: np.ndarray
            Pre-activation of the hidden layer of size (hidden_size),
            i.e., the input vector to the activation function.
        """
        z_h = np.dot(x, self.W_h) + self.b_h
        h = self.activation_func.forward(z_h)
        y = np.dot(h, self.W_o) + self.b_o
        return y, h, z_h
        
    def backward(self, x: np.ndarray, h: np.ndarray, z_h: np.ndarray, dloss: np.ndarray) -> dict[str, np.ndarray]:
        """
        Backward pass of the MLP.
        
        Parameters
        ----------
        x: np.ndarray
            Input vector of size (input_size).
        h: np.ndarray
            Activation of the hidden layer of size (hidden_size).
        z_h: np.ndarray
            Pre-activation of the hidden layer of size (hidden_size),
            i.e., the input vector to the activation function.
        dloss: np.ndarray
            Gradient of the loss function with respect to y_pred.
            
        Returns
        -------
        grads: dict
            Dictionary containing the elements:
            - "W_h": gradients for W_h
            - "b_h": gradients for b_h
            - "W_o": gradients for W_o
            - "b_o": gradients for b_o
        """
        b_o = dloss
        W_o = np.dot(h.T, dloss) 
        dloss = self.activation_func.backward(z_h).T * (self.W_o @ dloss)
        b_h = dloss
        W_h = np.dot(np.expand_dims(x, 1), dloss.T)

        return {"W_h": W_h, "b_h": b_h.T, "W_o": W_o, "b_o": b_o.T}
