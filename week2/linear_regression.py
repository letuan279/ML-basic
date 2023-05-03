import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

X, Y = datasets.load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]
TEST_SIZE = 20

X_train = X[:-TEST_SIZE]
X_test = X[-TEST_SIZE:]
Y_train = Y[:-TEST_SIZE]
Y_test = Y[-TEST_SIZE:]

reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

print("w = ", reg.coef_)

plt.scatter(X_test, Y_test, c='black')
plt.plot(X_test, Y_pred, c='blue', linewidth=3)

plt.show()
