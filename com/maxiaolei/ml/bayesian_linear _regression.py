from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
import numpy as np

X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

bayesModel=BayesianRidge(
                        n_iter=300,  #最大迭代次数
                        tol=1.e-3,   #停止训练的误差值大小
                        alpha_1=0.5,  #分布的形状参数？
                        alpha_2=0.5,  #比率参数
                        lambda_1=0.6,  #Gamma分布的形状参数
                        lambda_2=0.6,  #Gamma比例参数
                        compute_score=False,  #如果为真，则计算模型每一步的目标函数。
                        fit_intercept=True,   #是否计算结局
                        normalize=False,      #是否正则化
                        copy_X=True,          #X被复制？被覆盖
                        verbose=False         #详情模式
)

bayesModel.fit(X,y)


xx = np.linspace(5, 20, 100)
xx= xx.reshape(xx.shape[0],1)
yy=bayesModel.predict(xx)
plt.plot(X, y, 'k.')
plt.plot(xx, yy, 'r-')
plt.show()