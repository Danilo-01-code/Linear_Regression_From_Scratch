import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import logging


def shuffle_and_split(
        data: Union[pd.DataFrame, np.ndarray, Tuple[pd.DataFrame, pd.Series]],
        test_ratio:float = 0.2, 
        random_state:Optional[int] = None, 
        shuffle:bool = True
        ) -> Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
                   Union[pd.DataFrame, pd.Series],
                   Optional[Union[pd.Series, None]],
                   Optional[Union[pd.Series, None]]]:
    """
    Shuffle and split the data in a training and testing sets

    Parameters:
    ----------

    data (Union[pd.DataFrame, np.ndarray, Tuple[pd.DataFrame, pd.Series]]): 
        The dataframe or np array or a tuple with DataFrame and Series to be shuffled.
    test_ratio(float): The proportion of the dataset to include in the test split. This should be a value between 0 and 1.
    random_state(Int, Optional):Control the shuffling applied to the data before applying the split, for reproducibility.
    shuffle(bool, optional): Whether or not split the data before the spliting, Default is True.


    Returns:
    --------
    Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series], Optional[Union[pd.Series, None]], Optional[Union[pd.Series, None]]]
        - If 'data' is a Tuple: returns (X_train, X_test, y_train, y_test).
        - If 'data' is a DataFrame: returns (train_set, test_set).
    
    """
    #validing test ratio
    if test_ratio < 0 or test_ratio > 1:
        raise ValueError("variable test_ratio should be between 0 and 1")
    
    #Set random state if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    if isinstance(data, tuple):
        X, y = data
    else:
        if isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame):
            X, y = data, None
        else:
            raise ValueError("Data should be either a pd.DataFrame, np.ndarray or a tuple of (pd.DataFrame, pd.Series)")

    logging.info(f'X: {X}, y: {y}')

    if shuffle:
        index = np.random.permutation(len(X))
    else:
        index = np.arange(len(X))

    logging.info(f'Index order: {index}')

    test_size = int(len(X) * test_ratio)
    train_index = index[test_size:]
    test_index = index[:test_size]

    logging.info(f'test Index: {train_index}, test Index: {test_index}')

    if y is not None:
        # Handle DataFrame or Series for y
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:  # Assuming X is a np.ndarray
            X_train, X_test = X[train_index], X[test_index]

        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test
    
    else:
        #if y is not provided jjust return train and test sets of X
        if isinstance(X, np.npdarray):
            train_set = X.iloc[train_index]
            test_set = X.iloc[test_index]
        else: #Assumes data is a pd.DataFrame
            train_set = X[train_index]
            test_set = X[test_index]
        return train_set, test_set
        
    