# Packages
import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Loading Data
all_data = pd.read_csv(filepath_or_buffer='/input/calcofi/bottle.csv', usecols=["Salnty", "T_degC"])
lr_data = all_data[:300].dropna(axis=0, how='any') # dropping rows with null values
lr_data = lr_data.sample(frac=1, random_state=123).reset_index(drop=True) #random shuffle of the data

lr_train_data = lr_data[:int(len(lr_data)*0.5)]
lr_valid_data = lr_data[int(len(lr_data)*0.5):]

# Visualizing Training Data
plt.figure(figsize=(10,6))
plt.scatter(lr_train_data["T_degC"], lr_train_data["Salnty"], color='cornflowerblue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.show()


# Converting pandas DP to numpy array
x_train = lr_train_data['Salnty'].values.reshape(-1, 1) #-1 = all
y_train = lr_train_data['T_degC'].values.reshape(-1, 1)
x_valid = lr_valid_data['Salnty'].values.reshape(-1, 1)
y_valid = lr_valid_data['T_degC'].values.reshape(-1, 1)
aug_x = np.hstack([x_train, np.ones_like(x_train)])

# OLS + Plotting Functions
def ordinary_least_squares(x, y):
    xTx = x.T.dot(x)
    xTx_inv = np.linalg.inv(xTx)
    w = xTx_inv.dot(x.T.dot(y))
    return w
    
def polynomial(values, coeffs):
    assert len(values.shape) == 2
    # Coeffs are assumed to be in order 0, 1, ..., n-1
    expanded = np.hstack([coeffs[i] * (values ** i) for i in range(0, len(coeffs))])
    return np.sum(expanded, axis=-1)

def plot_polynomial(coeffs, x_range=[x_train.min(), x_train.max()], color='darkorange', label='polynomial', alpha=1.0):
    values = np.linspace(x_range[0], x_range[1], 1000).reshape([-1, 1])
    poly = polynomial(values, coeffs)
    plt.plot(values, poly, color=color, linewidth=2, label=label, alpha=alpha)
    
# Simple Linear Regression Plot
linear_coeff = ordinary_least_squares(aug_x, y_train)
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, color='cornflowerblue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plot_polynomial([linear_coeff[1,0], linear_coeff[0,0]])


def mean_squared_error(x, y, w):
    y_hat = x.dot(w)
    loss = np.mean((y - y_hat) ** 2)
    return loss

def avg_loss(x, y, w):
    y_hat = x.dot(w)
    loss = np.mean((y - y_hat) ** 2)
    return loss
    
linear_coeff = ordinary_least_squares(aug_x, y_train)
loss = avg_loss(aug_x, y_train, linear_coeff)

plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, color='cornflowerblue')
plt.xlabel("Temperature")
plt.ylabel("Salinity")
plt.title("Train Loss: {:.2f}".format(loss))
plot_polynomial([linear_coeff[1,0], linear_coeff[0,0]])



def polynomial_features(x, order):
    features = np.hstack([x**i for i in range(0, order+1)])
    return features

def plot_regression(x, y, degree):
    features = polynomial_features(x, degree)
    w = ordinary_least_squares(features, y)
    loss = mean_squared_error(features, y, w)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_train, y_train, color='cornflowerblue')
    plot_polynomial(w)
    plt.title("Polynomial degree: {}, Train MSE: {:.4f}".format(degree,loss))