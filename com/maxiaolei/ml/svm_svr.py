from sklearn.svm import  SVR
import numpy as np
import matplotlib.pyplot as plt

def runplt():
    plt.figure()
    plt.axis([0, 12, 0, 2])
    plt.grid(True)
    return plt

plt = runplt()
#随便定义一个生成Y的函数
def generateY(x):
    return x**2+3



X=np.linspace(1, 10, 10,dtype='int').reshape(10,1)
y=generateY(X)
yS=[[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]

y=np.array(y).reshape(10,1)


svr = SVR(C=0.1, kernel='sigmoid')
svr.fit(X,yS)
ymean=svr.predict(X)

print(svr.support_ )
print(svr.dual_coef_)
print(svr.intercept_)


plt.plot(X, yS,  'k.')
plt.plot(X, ymean, 'r-')
plt.show()