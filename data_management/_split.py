import numpy as np

def shuffle_and_split(data,test_ratio):
    """
    Shuffle and split the data in a training and testing sets

    Pramameters:
    ----------

    data (pandas.DataFrame): the dataset to be shuffle
    test_ratio(float): The proportion of the dataset to include in the test split. This should be a value between 0 and 1.

    Returns:
    --------

    tuple: a tuple containing two pandas df:
        -The Training set (pandas DataFrame)
        -The Testing set (pandas DataFrame)

    """
    shuffeled_indices = np.random.permutation(len(data))
    test_size = int(len(data) * test_ratio)
    test_indices = shuffeled_indices[:test_size]
    train_indices = shuffeled_indices[test_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
 