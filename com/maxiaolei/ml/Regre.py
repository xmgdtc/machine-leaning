# coding:utf-8
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV,LassoCV,ElasticNetCV,LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

alphas=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

#随便定义一个生成Y的函数
def generateY(x):
    return x**3-x**2+3
    #return x+2

X=np.linspace(1, 50, 50,dtype='int').reshape(50,1)
y=generateY(X)

y=np.array(y).reshape(50,1)
print(y)
#X,y都有了  y=x**3+3  x 从1到50

#线性最小二乘法
linearRegression=LinearRegression(
    fit_intercept=True,  #截距
    normalize=False,     #正则化
    copy_X=True,         #复制X 大概是如果对X正则化后可能会覆盖原来的X
    n_jobs=1             #CUP数量
    )
linearRegression.fit(X,y)
yLR=linearRegression.predict(X)

#岭回归
ridgeCV=RidgeCV(
    alphas=alphas,
    fit_intercept=True,
    normalize=False,
    scoring=None,              #自定义评分方式？
    cv=None,                   #自定义交叉验证？
    gcv_mode='auto',
    store_cv_values=False      #指示交叉验证值是否与之对应的标志
)

ridgeCV.fit(X,y)
yR=ridgeCV.predict(X)

#Lasso回归
lassoCV=LassoCV(
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    fit_intercept=True,
    normalize=False,
    precompute='auto', #预计算
    max_iter=1000,     #最大迭代次数
    tol=1e-4,
    copy_X=True,
    cv=None,
    verbose=False,
    n_jobs=1,
    positive=False,
    random_state=None,
    selection='cyclic'
)
lassoCV.fit(X,y)
yL=lassoCV.predict(X)

#ElasticNet回归 岭回归和lasso回归的结合体
elasticNetCV=ElasticNetCV(
    l1_ratio=1,   #L1和L2比例系数 越接近0越接近 岭回归 越接近1越接近lasso回归
    eps=1e-3,
    n_alphas=100,
    alphas=alphas,
    fit_intercept=True,
    normalize=False,
    precompute='auto',
    max_iter=1000,
    tol=1e-4,
    cv=None,
    copy_X=True,
    verbose=0,
    n_jobs=1,
    positive=False,
    random_state=None,
    selection='cyclic'
)
elasticNetCV.fit(X,y)
yE=elasticNetCV.predict(X)





quadraticFeaturizer = PolynomialFeatures(
    degree=2,  #阶数
    interaction_only=False,  #如果值为true(默认是false),则会产生相互影响的特征集。
    include_bias=True #是否包含偏差列
    )

X2=quadraticFeaturizer.fit_transform(X)

linearRegression2=LinearRegression(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=1
    )
linearRegression2.fit(X2,y)
yLR2=linearRegression2.predict(X2)

plt.plot(X, y,  'k.')
#plt.plot(X, yLR, 'r-') #线性
plt.plot(X, yLR2, 'm-') #2次线性
plt.plot(X, yR, 'y-')  #岭回归
plt.plot(X, yL, 'b-')  #lasso回归
plt.plot(X, yE, 'g-')  #elasticNet回归
plt.show()