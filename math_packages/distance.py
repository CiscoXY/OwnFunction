import numpy as np

def Mahalanobis_Distance(X):
    '''
    计算X的马氏距离,X应为n*p的矩阵,而不是数据框,p个变量,样本容量为n,返回一个1*p的向量存储dist
    '''
    n , m = X.shape
    S = np.cov(X.T) #* 求出样本协方差矩阵
    mu = np.average(X , axis = 0) #* 按列求均值
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = np.matmul(np.matmul(X[i] - mu , np.linalg.inv(S)) , (X[i] - mu).T)
    return dist

if __name__=='__main__':
    a = np.array([1,2,3,4])
    b = np.array([1,2,3,4])
    c = [np.array([1,2,3,5]) , [1,2,3] , [2,2]]
    print(a*b)
    print(c)