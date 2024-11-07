import logging
import numpy as np
from typing import Union, Optional, Tuple
import logging


def pearsons_correlation(
        y: np.ndarray,
        y_hat: np.ndarray,
        squared: bool = False) -> float:
    """
    Calculate Pearson's Correlation between (x & y).
    If det == True, calculate the Coefficient of determination R^2.

    Parameters:
    ----------
    
    y (Optional[str]): The real values (True Labels).
    y_hat (Optional[str]): The predicted values from the model.
    squared (bool): If True, returns R^2 instead of pearsons_correlation.

    Returns:
    --------
    float: Pearson's Correlation Coefficient between the variables 'x' and 'y'.
    """

    if len(y) < 2 or len(y_hat) < 2:
        logging.error("Columns must have at least two observations.")
        raise ValueError("Columns must have at least two observations.")
    
    sum_y = np.sum(y)
    sum_y_hat = np.sum(y_hat)
    sum_y_times_y_hat = np.sum(y * y_hat)
    sum_y_squared = np.sum(y ** 2)
    sum_y_hat_squared = np.sum(y_hat ** 2)

    n = len(y)  # Number of observations

    # Calculate the numerator and denominator for Pearson's correlation coefficient
    numerator = n * sum_y_times_y_hat - sum_y * sum_y_hat
    denominator = np.sqrt((n * sum_y_hat_squared - sum_y_hat ** 2) * (n * sum_y_squared - sum_y ** 2))

    if denominator == 0:
        logging.log(logging.ERROR, f"denominator = {denominator}, correlation is undefined")
        raise ZeroDivisionError("The denominator value is 0; the correlation is undefined.")
    
    r = numerator / denominator

    if squared:
        return r ** 2  # Return R^2
    else:
        return r  # Return Pearson's Correlation Coefficient

    
    
def MSE(y:np.ndarray, 
        y_hat:np.ndarray
        ) -> float:
    """
    Calculate the  Mean Squared Error (MSE)

    Parameters:
    -----------
    y (np.ndarray): the real values of y in the dataset
    y_hat (np.ndarray): the predicted values of y in the Hypothesis function

    Returns:
    ---------   
    float: the Standard Error
    """

    if y.shape != y_hat.shape:
        logging.error("y and y_hat must have the same lenght")
        raise ValueError("y and y_hat must have the same lenght")
    
    if y.size < 1 or y_hat.size < 1:
        logging.error("y and y_hat must be at least one in length")
        raise ValueError("y and y_hat must be at least one in length")

    #Calculate the residuals between predicted values(y_hat) and real values (y)
    residuals = y - y_hat

    #Calculate MSE
    mse = np.sum(residuals ** 2) / (len(y)-2)

    return mse
