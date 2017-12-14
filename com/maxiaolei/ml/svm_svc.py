# coding:utf-8

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
X= np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y = np.array([1,1,2,2])

'''
C：C-SVC的惩罚参数C C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
  　　0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features(特征数)
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出？
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0。
'''

clf = SVC(
            C=1.0,            #C-SVC的惩罚参数C
            kernel='rbf',     #默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
            degree=3,         #kernel=poly的情况下有效  函数的维度，默认是3
            gamma='auto',     #kernel= 'rbf', 'poly' and 'sigmoid' 下 kernel的系数  auto=1/特征数
            coef0=0.0,        #kernel='poly' 'sigmoid' 下 核函数中的独立项
            shrinking=True,   #是否开启启发式算法
            probability=False,#是否采用概率估计
            tol=1e-3,         #停止训练的误差值大小
            cache_size=1024,  #缓存大小
            class_weight=None,#类别的权重  dict字典 根据字典指定  balanced 使用y的值自动调整权重，与输入数据中的类频率成反比，例如n_samples /（n_classes * np.bincount（y））`
            verbose=False,    #详细输出  多线程不适用 选False基本没错就对了
            max_iter=-1,      #最大迭代次数。-1为无限制
            decision_function_shape='ovr',  #干什么用的？ ‘ovo’ 一对一, ‘ovr’ 多对多  or None 无, default=None
            random_state=None #用于概率估计的数据重排时的伪随机数生成器的种子
        )
clf.fit(X,y)


#我们先来简单的预测一下
Z=np.array([[-1,-2]])
y1=clf.predict(Z)
#Z[0]被分到了第一类
print(y1)