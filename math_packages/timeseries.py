import numpy as np
import pandas as pd

def Woldparameter(varphi , theta , length = 100):
    '''
    这个是对ARMA计算Wold系数的计算(提供ARMA模型的差分方程系数)
    传入两个参数数组,分别为Xt的差分方程系数和Zt的差分方程系数,长度分别为p和q,第一个元素都为1
    返回一个大小的数组储存Wold系数,默认长度为100
    '''
    if(varphi.ndim != 1 or theta.ndim != 1):
        print('the parameters array isn\'t 1d , please check and repeat')
        exit(-1)
    p = len(varphi) - 1; q = len(theta) - 1
    if(p < 0 or q < 0):
        print('the parameters array\'s len is 0 , please check and repeat')
        exit(-1)
    phi = np.array([1.]) #* 最后结果的初始化
    if(p == 0 and q == 0):
        return np.append(phi , np.zeros(length - 1))
    elif( p == 0 and q):
        return theta
    for j in range(1 , length):
        if(j > q):
            b_j = 0
        else:
            b_j = theta[j]
        if(j <= p):
            phi_j = b_j + np.matmul(varphi[1:j+1] , phi[::-1]) 
        else:
            phi_j = b_j + np.matmul(varphi[1:] , np.flip(phi[j-p:]))
        phi = np.append(phi , phi_j)
    return phi
def AutoCovariance(varphi , theta , sigma2 , n = 100 , j_max = 100):
    '''
    varphi:控制Xt的差分方程的参数
    theta:控制Zt的差分方程的参数
    sigma2:白噪声的方差
    n:希望的自协方差函数的阶数(实际上返回的矩阵为n+1阶)    j_max:Wold系数的长度,指标从0到j_max-1
    '''
    if(n<=0 or j_max<=0):
        print('n,j_max must > 0')
        exit(-1)
    n+=1
    Woldparams = Woldparameter(varphi , theta , length = n+j_max)
    gamma_n = np.array([])
    for i in range(n):
        gamma_n = np.append(gamma_n , np.matmul(Woldparams[:j_max] , Woldparams[i:i+j_max]))
    gamma_n = gamma_n * sigma2 #*乘上方差
    Gamma = np.array([gamma_n])
    for i in range(1,n):
        temp = np.append( np.flip(gamma_n[1:i+1]), gamma_n[:n-i])
        Gamma = np.append(Gamma , np.array([temp]) , axis=0)
    return Gamma

def NewMessage_parameters(Gamma):
    '''
    输入协方差矩阵,计算对应维数的新息系数(包含所有迭代过程中的新息系数)
    '''
    gamma_n = Gamma[0]#* 获得对应的自协方差函数向量
    n = len(gamma_n)
    v_n = np.array([gamma_n[0]])
    theta = [np.array([gamma_n[1]/gamma_n[0]])]#*初始化theta
    v_n = np.append(v_n ,gamma_n[0] - theta[0][0]**2 * v_n[0])
    for i in range(2,n):
        theta_i_vector = np.array([])
        for k in range(i):
            if(k == 0):
                theta_i_imink = gamma_n[i-k]/v_n[k]
            else:
                theta_i_imink = (gamma_n[i-k] - sum(theta[k-1] * theta[-1][len(theta) - k : ] * np.flip(v_n[:k]) ))/v_n[k]
            theta_i_vector = np.append(theta_i_vector , theta_i_imink) #*添加到这个向量当中
        theta_i_vector = theta_i_vector[::-1] #* 反转
        theta.append(theta_i_vector)
        
        v_n = np.append(v_n , gamma_n[0] - np.matmul(np.flip(theta[-1]**2) , v_n))
    return theta #*把完整的迭代的theta作为结果传回去

def NewMessage_predict(X , Gamma , h):
    '''
    输入已经观测到的时间序列,dim=1xn
    Gamma:输入的协方差矩阵 
    h:h步预报
    '''
    if(X.ndim != 1): 
        print('the dim of X must be 1 , please repeat')
        exit(-1)
    if(h<1):
        print('h must >= 1')
        exit(-1)
    n_G , m_G = Gamma.shape ; n = len(X)
    if(n_G != m_G):
        print('Gamma\'s dim isn\'t , please check')
        exit(-1)
    elif(n_G < n + h):
        print('Dim(Gamma) must >= len(X) + h')
        exit(-1)
    CovMatrix = Gamma[:n+h+1][:,:n+h+1] #* 获得n+h,n+h的协方差矩阵
    theta = NewMessage_parameters(CovMatrix)#* 获得对应的新息系数
    X_h = np.array([])#* 创建最后的h步预报的向量
    for i in range(1 , h+1)
        for j in range(n):
            if(j == 0)
                X_jplusi = 0 
            else:
                X_jplusi = 
            X_h = np.append(X_h , X_jplusi)

if __name__ == "__main__":
    a = np.array([1,0.5,0.28])
    b = np.array([1])
    G = AutoCovariance(a , b , 1 , n=50 , j_max=50)
    
    print(NewMessage_parameters(G)[-1])
    