import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


#使用iris数据
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
print(X)
y = iris.target

# 训练模型
clf = DecisionTreeClassifier(
                 criterion="entropy",  #gini 基尼系数 entropy 信息增益
                 splitter="best",   #best 所有划分点寻找最优，数据量不大 randon随机的在部分划分点中找局部最优的划分点，数据量巨大
                 max_depth=None, #最大深度
                 min_samples_split=2, #内部节点再划分所需最小样本数
                 min_samples_leaf=1, #叶子节点最少样本数
                 min_weight_fraction_leaf=0., #叶子节点最小的样本权重和
                 max_features=None,  #划分时考虑的最大特征数   auto max_features = sqrt（n_features） sqrt max_features = sqrt（n_features） log2 max_features = log2（n_features）`。
                 random_state=None,
                 max_leaf_nodes=None,  #最大叶子节点数
                 min_impurity_decrease=0.,
                 min_impurity_split=None,  #节点划分最小不纯度
                 class_weight=None, #类别权重
                 presort=False  #数据是否排序
                 )
#拟合模型
clf.fit(X, y)


# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()
