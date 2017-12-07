from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X, y = iris.data[:,[0,2]], iris.target


#sample = [[6, 4, 5.5, 2],[5,3,2, 1]]
sample = [[6, 4],[5,3]]


knn = KNeighborsClassifier(
                n_neighbors=10,  #判断类型的紧邻数
                weights='distance', #距离权重  uniform无权重  distance距离的倒数  function Array=>Array
                algorithm='auto',  #算法 'ball_tree', 'kd_tree', 'brute'
                leaf_size=1000, #叶子节点大小
                p=2,          #Power parameter for the Minkowski metric
                metric='minkowski', #默认“minkowski”  用于树的距离度量
                metric_params=None,   #度量函数的其他关键字参数。 可选参数 默认无
                n_jobs=-1,  #并行数量
                )
knn.fit(X, y)
predicted_value = knn.predict(sample)
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))


# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()