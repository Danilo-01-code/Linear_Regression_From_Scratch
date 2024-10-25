"""
Linear Regression using Sthocastic Gradient Descent or Bath Gradient Descent
"""

import numpy as np
import logging

class LinearRegression:
    """
    Implements simple Linear Regression

    Parameters:
    ----------
    lr : float, opcional (default=0.01)
        TLearning Rate
    n_iters : int, opcional (default=1000)
        Number of Iterations

    Methods:
    -------
    fit(X, y)
        Fits the model to the feature matrix X and labels y.
    predict(X)
        Predicts the labels for feature matrix X.
    """

    
    def __init__(self,l_r=0.01, n_iters=1000):
        self.l_r = l_r
        self.n_iters = n_iters

        self.intercept = 0
        self.coef = 1
        logging.info("Model Initialized.")
    

    def SGD(self):
        #SGD (Sthocastic Gradient Descent)
        #In each iteration, the gradient is calculated using only a single example from the training dataset.

        #Random selection of a subset of data
        idx = np.random.choice(len(self.x), self.sample_size, replace = False)
        x_sample = self.x[idx]
        y_sample = self.y[idx]

        #Sample Predict
        y_predict = self.intercept + self.coef * x_sample

        Gradient_Intercept, Gradient_coef = self.partial_derivative(np.array([y_predict]), np.array([y_sample]),x_sample)

        #Update Parameters
        self.intercept -= self.l_r * Gradient_Intercept
        self.coef -= self.l_r * Gradient_coef

        logging.debug(f"Gradient Intercept: {Gradient_Intercept:.4f}, Gradient_coefficient {Gradient_coef:.4f}")


    def BGD(self):
        #Bath Gradient Descent
        #In each iteration, we calculate the gradient of the cost function using all examples from the training dataset.

        #Predict for all the data
        y_predict = self.intercept + self.coef * self.x
        
        Gradient_Intercept, Gradient_coef = self.partial_derivative(y_predict, self.y, self.x)
    
        # Update the parameters
        self.intercept -= self.l_r * Gradient_Intercept
        self.coef -= self.l_r * Gradient_coef


    def partial_derivative(self, y_predict, y_sample,x_sample):
        """
        Calculates the Partial Derivative of the MSE with respect to the intercept and coefficient.
        
        Parameters
        -----
        y_predict: array-like, shape: (1,)
            The predicted value for a single sample.
        
        y_sample: array-like, shape: (1,)
            The actual value for a single sample.
        
        Returns
        -----
        Gradient_Intercept: float
            The partial derivative of the MSE with respect to the intercept.
        
        Gradient_coef: float
            The partial derivative of the MSE with respect to the coefficient.
        """

        #Calculate the error for all samples
        error = y_sample - y_predict

        #Calculate the gradients
        Gradient_Intercept = (-2 / len(y_sample)) * np.sum(error)  # Derivative of MSE w.r.t. intercept
        Gradient_coef = (-2 / len(y_sample)) * np.sum(error * x_sample)  # Derivative of MSE w.r.t. coefficient

        return Gradient_Intercept, Gradient_coef



    def fit(self, x: np.ndarray, y: np.ndarray, sample_size=10, SGD: bool = True):
        """
        Fit a Linear Model to the given data.

        Parameters
        ----------
        x: np.ndarray
            Array of shape (n_samples, n_features) representing the input features.

        y: np.ndarray
            Array of shape (n_samples,) representing the target values.

        sample_size: int, optional, default=10
            The number of samples to use for each SGD update. Relevant only if SGD is True.

        SGD: bool, optional, default=True
            If True, use Stochastic Gradient Descent for training the model. 
            If False, use Batch Gradient Descent.

        Returns
        -------
        self: object
            Returns the instance of the fitted model for chaining.

        Notes
        -----
        - Ensure that the input arrays x and y have compatible shapes.
        - The model parameters will be updated iteratively based on the specified gradient descent method.
        """

        self.sample_size = sample_size
        self.x = np.asarray(x) 
        self.y = np.asarray(y)

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Input features and target values must have the same length.")
        logging.info("Starting training Process...")

        for i in range(self.n_iters):
            if SGD:
                self.SGD() # Pass sample_size to SGD method
            else:
                self.BGD() # Assume you have a BGD method defined

            if i % 10 == 0: #Log for every 10 iterarions
                loss = np.mean((self.y - self.predict(self.x))**2) #MSE (Mean Squared Error)

                logging.info(f"Iteration {i}: Loss = {loss:.4f}")

        logging.info("Training Completed.")
        return self
    
    #TODO _coef and _intercept
    #TODO __Str__ method
    def predict(self, x:np.ndarray) -> np.ndarray:
        return self.intercept + self.coef * x #Hypothesis function