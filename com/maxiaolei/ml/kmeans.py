
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans,MiniBatchKMeans
#这里生成了一些样本 X为样本特征，Y为样本簇类别
X, y = make_blobs(
    n_samples=1000, #样本数量
    n_features=2,  #样本维度数
    centers=[[-1,-1], [0,0], [1,1], [2,2]], #中心点
    cluster_std=[0.5, 0.2, 0.3, 0.2],       #方差
    shuffle=True,
    random_state =1
    )
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()




kmeans_model = KMeans(n_clusters=4, #分类簇群的数量
                init='k-means++', #初始选点方法'k-means++', 'random' or an ndarray
                n_init=10, #每个点的运行次数
                max_iter=300, #最大迭代次数
                tol=1e-4, #容忍度？
                precompute_distances='auto', #预计算 需要更大内存
                verbose=0,  #详情模式？
                random_state=None, #随机发生器？
                copy_x=True, #precompute_distances=true的时候选false会改变原始数据？
                n_jobs=-1,  #并行数量  -1为全部cpu
                algorithm='auto' #使用的算法 auto full elkan
                )

#对大样本优化的Mini Batch K-Means
#通过 进行多次的 无放回随机抽样采一部分样本batch_size 求最优解
min_kmeans_model=MiniBatchKMeans(n_clusters=4,  #分类簇群的数量
                                 init='k-means++',  #初始选点方法'k-means++', 'random' or an ndarray
                                 max_iter=300,  #最大迭代次数
                                 batch_size=100,  #批量的大小
                                 verbose=0,  #详情模式？
                                 compute_labels=True,
                                 random_state=None,
                                 tol=0.0,
                                 max_no_improvement=10,
                                 init_size=4,  #随机抽样的样本数量
                                 n_init=4,  #随机初始化的数量
                                 reassignment_ratio=0.01
                                 )

kmeans=min_kmeans_model

y_pred =kmeans.fit_predict(X)

#用下面这个来评估聚类的模型的分数
from sklearn import metrics
score=metrics.calinski_harabaz_score(X, y_pred)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (4, score)),
         transform=plt.gca().transAxes, size=10,
         horizontalalignment='right')
plt.show()

