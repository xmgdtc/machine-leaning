import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
def runplt():
    plt.figure()
    plt.axis([5, 20, 5, 20])
    plt.grid(True)
    return plt

plt = runplt()
#模拟数据
X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]

# 创建并拟合模型
model = LinearRegression(
    fit_intercept=True, #是否计算截据 似乎false就是从一个点作为起始点 而true则会计算一个y值 起点在起始点的y1+y
    normalize=True,   #是否归一化
    copy_X=True,    #X被复制？
    n_jobs=-1        #cpu数量
    )
#训练模型
model.fit(X, y)
#预测
y2 = model.predict(X)

#增加残差值
#for idx, x in enumerate(X):
#    plt.plot([x, x], [y[idx], y2[idx]], 'r-')

#残差平方和 越小越好
rsos=np.mean((model.predict(X) - y) ** 2)

plt.text(.99, .01, ('rsos=%f' %rsos),
         transform=plt.gca().transAxes, size=10,
         horizontalalignment='right')
#展现
plt.plot(X, y, 'k.')
plt.plot(X, y2, 'g-')

#还不知道这段干嘛的 但是通过他生成了多阶方程？
quadratic_featurizer = PolynomialFeatures(
    degree=2,  #阶数
    interaction_only=False,  #如果值为true(默认是false),则会产生相互影响的特征集。
    include_bias=True #是否包含偏差列
    )

'''
PolynomialFeatures方法
fit（X [，y]）	训练生成模型。
fit_transform（X [，y]）	转换数据 然后转换它。 阶次跟degree有关 如2次的 2=>[1,2,4] 3=>[1,3,9]   [2,3]=>[[1,2,4],[1,3,9]]
get_feature_names（[input_features]）	返回输出要素的特征名称
get_params（[深]）	获取此估算器的参数。
set_params（** PARAMS）	设置此估算器的参数。
transform（X）	将数据转换为多项式特征
'''
#将原始数据变成三元的，并且训练他 得到三元的回归模型regressor_quadratic
#生成了多阶的数组 2=>[1,2,4] 3=>[1,3,9]   [2,3]=>[[1,2,4],[1,3,9]]
X_train_quadratic = quadratic_featurizer.fit_transform(X)
regressor_quadratic = LinearRegression(fit_intercept=True,normalize=True,  copy_X=True, n_jobs=-1  )
#根据生成的多阶数组训练  模拟多元 原来是1个参数 现在是3个了  我真是机智
regressor_quadratic.fit(X_train_quadratic, y)

#下面就是建立了一个测试数据的x，并把它变成三元的  ，之后通过三元的模型去回归预测yy
xx = np.linspace(5, 20, 100)  #在0到20之前均匀的生成100个数 [ 5,  5.1  ,5.2] 类似
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))  #老套路 将预测的x值也变成三元的
yy=regressor_quadratic.predict(xx_quadratic) #然后根据三个参数预测出y的值来

#虽然预测是三元的，但是我们还拿一元的去划线 这样就出现曲线图了
plt.plot(xx, yy, 'r-')
plt.show()

