# Linear Regression from Scratch
This repository contains an implementation of linear regression from scratch using Numpy. It includes both Stochastic Gradient Descent (SGD) and Batch Gradient Descent (BGD) optimization techniques, and Matplotlib for result visualization.

## Project Objective
The objective of this project is to provide a clear, detailed, and accessible implementation of linear regression to help students and professionals better understand the underlying workings of machine learning models.

By focusing on transparency, this project aims to:

Explain core Linear Regressions concepts in a hands-on manner.
Offer detailed annotations and docstrings for each function and calculation.
Provide a step-by-step approach to building a predictive model from scratch.
This project is perfect for anyone looking to enhance their understanding of linear regression and deepen their knowledge of machine learning fundamentals.

## Getting Started
To run this project locally, you need to install the dependencies. All requirements are listed in the requirements.txt file.

## Prerequisites
To run this project you need install the necessary dependencies by running:


`pip install -r requirements.txt`

## Features
- **Stochastic Gradient Descent (SGD)**: Optimizes the model using mini-batches for faster convergence.
- **Batch Gradient Descent (BGD)**: Optimizes the model using the full dataset for a more precise but slower convergence.
- **Visualization**: Plots the data and regression lines using Matplotlib to visualize the model’s performance.
- **shuffle_and_split**: A function that shuffles the dataset and splits it into training and testing sets, ensuring randomness for model evaluation.
- **Pearson's Correlation**: A metric to evaluate the linear correlation between the predicted and actual values. Ranges from -1 (perfect negative correlation) to 1 (perfect positive correlation).
- **R² (Coefficient of Determination)**: A metric that indicates how well the model's predictions match the actual data. Ranges from 0 (no correlation) to 1 (perfect correlation).
- **Mean Squared Error (MSE)**: A metric that measures the average squared difference between the predicted and actual values. A lower MSE indicates a better fit of the model to the data.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
