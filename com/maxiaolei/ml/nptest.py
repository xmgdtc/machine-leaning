import numpy as np
array=np.array([1,2,3,3],dtype=float) #生成np数组
a= np.ones([2,3], dtype=np.int32)  #生成全部为1的数组
b= np.linspace(0,15,3)            #从0到15 生成3个等差数组
d=np.arange(10, 30, 5)           #从10(包含)到30(不包含) 生成n个等差为5的数组
d1=np.arange(4)                 #从0到4(不包含) 生成n个等差为1的数组
print(b)
print(d)
print(b.sum())  #求合
print(b.min())  #最小
print(b.max())  #最大
print('array.mean',array.mean()) #平均
print(array[1:3])
print(array[::-1])


c=np.array([[1,2,3],[4,5,6,7]])
print("'''''''''''''''''''''")
print(c)
print('维数',c.ndim)#：数组的维数（即数组轴的个数），等于秩。最常见的为二维数组（矩阵）。

print('维度',c.shape)#：数组的维度。为一个表示数组在每个维度上大小的整数元组。例如二维数组中，表示数组的“行数”和“列数”。ndarray.shape返回一个元组，这个元组的长度就是维度的数目，即ndim属性。

print('数组大小',c.size)#：数组元素的个数，等于shape属性中元组元素的乘积。

print(c.dtype)#：表示数组中元素类型的对象，可使用标准的Python类型创建或指定dtype。另外也可使用前一篇文章中介绍的NumPy提供的数据类型。

print(c.itemsize)#：数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为8(float64占用64个bits，每个字节长度为8，所以64/8，占用8个字节），又如，一个元素类型为complex32的数组item属性为4（32/8）。

print(c.data)#：包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。