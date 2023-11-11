import numpy as np
from typing import List, Tuple

class GradientDescent():

    @staticmethod
    def evaluate(data: List[Tuple[np.ndarray, np.ndarray]], model, loss_func) -> List[float]:
        """
        function to evaluate the test data
        i.e., just forward pass and loss computation
        
        Parameters
        ----------
        data:
            input data containing X and y
        model:
            the initialized MLP model
        loss_func:
            loss function of your choice
        
        Returns
        -------
        losses:
            array containing all individual losses
            i.e., for each data sample
        """
        losses = []
        for x, y_true in data:
            y_pred, h, z_h = model.forward_(x)
            loss = loss_func.forward(y_true, y_pred)
            losses.append(loss)
        return losses
    
    @staticmethod
    def update(data: List[Tuple[np.ndarray, np.ndarray]], model, loss_func, learning_rate: float) -> List[float]:
        """
        function to calculate gradients and perform weight updates
        i.e., forward pass + loss computation + backward pass + weight update
        
        Parameters
        ----------
        data:
            input data containing X and y
        model:
            the initialized MLP model
        loss_func:
            loss function of your choice
        learning_rate:
            float value defining the learning rate
        
        Returns
        -------
        losses:
            array containing all individual losses
            i.e., for each data sample
        """
        losses = []
        for x, y_true in data:
            y_pred, h, z_h = model.forward_(x)
            loss = loss_func.forward(y_true, y_pred)
            losses.append(loss)

            dloss = loss_func.backward(y_pred, y_true)
            grads = model.backward(x, h, z_h, dloss)

            model.W_h -= learning_rate * grads["W_h"]
            model.b_h -= learning_rate * grads["b_h"]
            model.W_o -= learning_rate * grads["W_o"]
            model.b_o -= learning_rate * grads["b_o"]
        
        return losses
    
    def gradient_descent(self, train_data, test_data, model, loss_func, epochs, learning_rate):
        valid_losses = self.evaluate(test_data, model, loss_func)
        print("Initial Validation: " + str(np.mean(valid_losses)))
        
        train_loss_arr = []
        valid_loss_arr = []
        
        for epoch in range(epochs):
            train_losses = self.update(train_data, model, loss_func, learning_rate)
            valid_losses = self.evaluate(test_data, model, loss_func)
            
            print("Epoch " + str(epoch) + ": " + str(np.mean(train_losses)) + " Train Loss, " + str(np.mean(valid_losses)) + " Valid Loss") 
            
            train_loss_arr.append(np.mean(train_losses))
            valid_loss_arr.append(np.mean(valid_losses))
            
        return train_loss_arr, valid_loss_arr