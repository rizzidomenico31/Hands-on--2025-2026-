import operator
import pandas as pd
import numpy as np

from evaluation import Evaluation
from linear_regression import LinearRegression
import matplotlib.pyplot as plt

# read the dataset
houses = pd.read_csv('datasets/houses.csv')

# shuffling all samples to avoid group bias
houses = houses.sample(frac=1).reset_index(drop=True)

# select only some features, also you can try with other features
x = houses[['GrLivArea', 'LotArea']]

# select target value
y = houses['SalePrice']

# print the correlation of features
print(x.corr())

x = x.values
y = y.values

x_square = x**2
x_cubic = x**3
x_4 = x**4
x = np.column_stack((x, x_square, x_cubic, x_4))

# split dataset into training and test
train_index = round(len(x) * 0.8)

X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# normalize both sets using the statistics computed on training data
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# add the bias column
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# instantiate a linear regression object specifying the parameters needed
linear = LinearRegression(n_features=X_train.shape[1], n_steps=1000, learning_rate=0.05)

cost_history, theta_history = linear.fit_fullbatch(X_train, y_train)

print(f'''Thetas: {*linear.theta,}''')
print(f'''Final train cost/MSE:  {cost_history[-1]:.3f}''')

pred = linear.predict(X_test)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test[:, 1], pred), key=sort_axis)
x_poly, y_poly_pred = zip(*sorted_zip  )

plt.plot(X_test[:, 1], y_test, 'r.', label='Training data')
plt.plot(x_poly, y_poly_pred, 'b--', label='Current hypothesis')
plt.legend()
plt.show()
plt.show()

eval = Evaluation(linear)

print(eval.compute_performance(X_test, y_test))