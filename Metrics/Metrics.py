import pandas as pd
from math import sqrt
import logging
import numpy as np

def pearsons_correlation(p: pd.DataFrame, x_col: str, y_col: str, det: bool = False) -> float:
    """
    Calculate Pearsons Correlation between (x & y)
    if det == True 
    Calculate the Coefficient of determination R^2

    Parameters:
    ----------

    p (pd.DataFrame): DataFrame containing at least two columns named 'x' and 'y' 
                      representing the variables to correlate.

    Returns:
    --------

    float: Pearson's Correlation Coefficient between the variables 'x' and 'y'.
    """

    if x_col not in p.columns or y_col not in p.columns:
        logging.error(f"x: {x_col} or y {y_col} not in Dataframe")
        raise ValueError(f"x: {x_col} or y {y_col} not in Dataframe")
    
    logging.info(f"Selected columns {x_col} and {y_col}")

    #Extrating variables x and y
    x = p[x_col]
    y = p[y_col]

    if len(x) < 2 or len(y) < 2:
        logging.error("Columns must have at least two obsercations.")
        raise ValueError("Columns must have at least two obsercations.")
    
    n = len(p) #Observation Numbers

    #Sum of squares calculation
    sum_x = sum(x)
    sum_xy = sum(x*y)
    sum_y = sum(y)
    sum_x_squared = sum(x ** 2)
    sum_y_squared = sum(y ** 2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = sqrt(n * sum_x_squared - sum_x**2) * sqrt(n * sum_y_squared - sum_y ** 2)

    r = numerator / denominator

    if det:
        return r**2
        # r^2 == 1 , The model explains 100% of the variation in data.
        # r^2 == 0 , the model explains None of the variation. 
    else:
        # r == 0 , no correlation.
        # r > 0 , positive correlation.
        # r < 0 , negative correalation.
        return r
    
    
def MSE(y:np.ndarray, y_hat:np.ndarray) -> float:
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
