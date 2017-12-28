


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


def runplt():
    plt.figure()
    plt.axis([5, 20, 5, 20])
    plt.grid(True)
    return plt



X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 建立线性回归，并用训练的模型绘图
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt = runplt()
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_featurizer.fit_transform(X_train)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))



quadratic_featurizer3 = PolynomialFeatures(degree=3)
X_train_quadratic3 = quadratic_featurizer3.fit_transform(X_train)

regressor_quadratic3 = LinearRegression()
regressor_quadratic3.fit(X_train_quadratic3, y_train)
xx_quadratic3 = quadratic_featurizer3.transform(xx.reshape(xx.shape[0], 1))

quadratic_featurizer4 = PolynomialFeatures(degree=4)
X_train_quadratic4 = quadratic_featurizer4.fit_transform(X_train)

regressor_quadratic4 = LinearRegression()
regressor_quadratic4.fit(X_train_quadratic4, y_train)
xx_quadratic4 = quadratic_featurizer4.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.plot(xx, regressor_quadratic3.predict(xx_quadratic3), 'b-')
plt.plot(xx, regressor_quadratic4.predict(xx_quadratic4), 'g-')
plt.show()



print(regressor.score(X_train, y_train))
print(regressor_quadratic.score(X_train_quadratic, y_train))
print(regressor_quadratic3.score(X_train_quadratic3, y_train))
print(regressor_quadratic4.score(X_train_quadratic4, y_train))


from sklearn import model_selection
model_selection.train_test_split()