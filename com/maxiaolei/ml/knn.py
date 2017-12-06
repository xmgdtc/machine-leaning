from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X, y = iris.data, iris.target


sample = [[6, 4, 5.5, 2],[5,3,2, 1]]

# k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)
predicted_value = knn.predict(sample)
print('k=10')
#分类
print(iris.target_names[predicted_value])
#各分类下概率
print(knn.predict_proba(sample))

#k=10 按距离加权
knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(X, y)
predicted_value = knn.predict(sample)
print('k=10 and weights=distance')
print(iris.target_names[predicted_value])
print(knn.predict_proba(sample))
