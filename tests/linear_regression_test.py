"""
Linear Regression test
"""

import numpy as np
import matplotlib.pyplot as plt
from ..Linear_model.LinearRegression import LinearRegression
import pandas as pd
from ..Metrics.Metrics import pearsons_correlation, MSE
from ..data_management._split import shuffle_and_split 

np.random.seed(40)

n_samples = 100

X = 2 * np.random.rand(n_samples,1)

slope = 3
intercept = 4

noise = np.random.randn(n_samples,1)
y = slope * X + intercept + noise

model = LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)

#TODO shuffle and split the data
#TODO metrics with pearson correlation and MSE

# Visualizando os resultados
plt.scatter(X,y, color='blue', label='Random Data')
plt.plot(X, y_pred, color='red', linewidth=2, label= 'Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression From Scrath')
plt.legend()
plt.grid()
plt.show()

