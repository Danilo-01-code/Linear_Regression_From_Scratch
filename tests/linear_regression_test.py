"""
Linear Regression test
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from Linear_model.LinearRegression import LinearRegression
from Metrics.Metrics import pearsons_correlation, MSE
from data_management.split import shuffle_and_split 

# Defining seed for reproducibility
np.random.seed(40)

# Generating Simulated data
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)
slope = 3
intercept = 4
noise = np.random.randn(n_samples, 1)
y = slope * X + intercept + noise

#shuffle and split the data
X_train, X_test, y_train, y_test = shuffle_and_split((X, y), test_ratio = 0.2, random_state = 42)

# Instantiating the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting Values
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
train_corr = pearsons_correlation(y_train, y_pred_train)
test_corr = pearsons_correlation(y_test, y_pred_test)
train_mse = MSE(y_train, y_pred_train)
test_mse = MSE(y_test, y_pred_test)

print(f"Train Pearson Correlation: {train_corr:.3f}")
print(f"Test Pearson Correlation: {test_corr:.3f}")
print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")

# Ploting Results
plt.scatter(X, y, color='blue', label='Random Data')
plt.plot(X_train, y_pred_train, color='green', linewidth=2, label='Train Regression Line')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Test Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression From Scratch')
plt.legend()
plt.grid()
plt.show()
