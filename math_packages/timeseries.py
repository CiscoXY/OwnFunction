import numpy as np
import pandas as pd
from scipy.stats import f # F分布相关的api
from scipy.stats import norm # 标准正太的api
from scipy.stats import chi2 # 卡方分布的api
from scipy.stats import t # t分布的api
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
def Greenparameters(varphi , theta , length = 100):
    '''
    这个是对ARMA计算Green系数的计算(提供ARMA模型的差分方程系数)
    传入两个参数数组,分别为Xt的差分方程系数和Zt的差分方程系数,长度分别为p和q,第一个元素都为1
    返回一个大小的数组储存Green系数,默认长度为100
    '''
    if(varphi.ndim != 1 or theta.ndim != 1):
        print('the parameters array isn\'t 1d , please check and repeat')
        exit(-1)
    p = len(varphi) - 1; q = len(theta) - 1
    if(p < 0 or q < 0):
        print('the parameters array\'s len is 0 , please check and repeat')
        exit(-1)
    Green = np.array([1.]) #* 最后结果的初始化
    if(p == 0 and q == 0):
        return np.append(Green , np.zeros(length - 1))
    elif( p == 0 and q):
        if(length - len(theta) < 0):
            return np.append(Green , -theta[1:length])
        else:
            return np.append(np.append(Green , -theta[1:]) , np.zeros(length - len(theta)))
    for k in range(1 , length):
        if(k>q):
            theta_k = 0
        else:
            theta_k = theta[k]
        if(k <= p):
            Green = np.append(Green , np.matmul(varphi[1:k+1] , np.flip(Green)) - theta_k)
        else:
            Green = np.append(Green , np.matmul(varphi[1:] , np.flip(Green[k-p:])) - theta_k)
    return Green
def Inverseparameters(varphi , theta , length = 100):
    '''
    这个是对ARMA计算逆函数系数的计算(提供ARMA模型的差分方程系数)
    传入两个参数数组,分别为Xt的差分方程系数和Zt的差分方程系数,长度分别为p和q,第一个元素都为1
    返回一个大小的数组储存逆函数系数,默认长度为100
    '''
    if(varphi.ndim != 1 or theta.ndim != 1):
        print('the parameters array isn\'t 1d , please check and repeat')
        exit(-1)
    p = len(varphi) - 1; q = len(theta) - 1
    if(p < 0 or q < 0):
        print('the parameters array\'s len is 0 , please check and repeat')
        exit(-1)
    Inverse = np.array([1.]) #* 最后结果的初始化
    if(p == 0 and q == 0):
        return np.append(Inverse , np.zeros(length - 1))
    elif( q == 0 and p):
        return np.append(np.append(Inverse , -varphi[1:]) , np.zeros(length - len(varphi)))
    for k in range(1 , length):
        if(k>p):
            varphi_k = 0
        else:
            varphi_k = varphi[k]
        if(k <= q):
            Inverse = np.append(Inverse , np.matmul(theta[1:k+1] , np.flip(Inverse)) - varphi_k)
        else:
            Inverse = np.append(Inverse , np.matmul(theta[1:] , np.flip(Inverse[k-q:])) - varphi_k)
    return Inverse
def LinearPredict(X , varphi , theta , h , sigma2 = 1 , mu = 0.0):
    '''
    输入已经观测到的时间序列,dim=1xn
    varphi  theta:输入差分方程系数 
    sigma2:白噪声的方差 默认为1 ; mu:初始偏差也就是Xt = mu + phi1Xt-1 ………… + epsillont + theta1*epsillont-1
    h:h步预报
    并返回Xt+1~X_t+h的预报以及X_t+1~X_t+h的均方误差
    '''
    if(X.ndim != 1): 
        print('the dim of X must be 1 , please repeat')
        exit(-1)
    if(h<1):
        print('h must >= 1')
        exit(-1)
    t = len(X) ;    temp = X
    I = Inverseparameters(varphi , theta , length = t+1) #* 获得从I0 ~ In的逆系数
    G = Greenparameters(varphi , theta , length = h)
    I = -I[1:] #* 选取-I1 , -I2 , ………… , -It
    X_prdict=  np.array([]) ; Var_error = np.array([])
    for i in range(1 , h+1):
        X_tplusi = mu + np.matmul(I , np.flip(temp[-t:]))
        X_prdict = np.append(X_prdict , X_tplusi)
        temp = np.append(temp , X_tplusi)
    for i in range(1 , h+1):
        Var_error_i = sigma2 * np.sum(G[:i]**2)
        Var_error = np.append(Var_error , Var_error_i)
    return X_prdict , Var_error
def Confidence_Region(X , varphi , theta , h , sigma2 = 1 , mu = 0.0 , alpha = 0.05):
    '''
    输入已经观测到的时间序列,dim=1xn
    varphi  theta:输入差分方程系数 
    sigma2:白噪声的方差 默认为1
    alpha :显著性水平,默认为0.05
    h:h步预报
    并返回X_t+1 ~ X_t+h的置信区间
    '''
    if(X.ndim != 1): 
        print('the dim of X must be 1 , please repeat')
        exit(-1)
    if(h<1):
        print('h must >= 1')
        exit(-1)
    X_pridict , Var_error = LinearPredict(X , varphi , theta , h , sigma2 = sigma2 , mu=mu)
    Index = [] ; t = len(X) ; Z = norm.ppf(1-alpha/2 , 0 , 1) #* 获得标准正态分布的左侧1-alpha/2分位数
    for i in range(1 , h+1):
        Index.append('x%d+%d'%(t , i)) #* 构造最后的index
    df = pd.DataFrame({'置信下限':X_pridict - Z * np.sqrt(Var_error) , '置信上限':X_pridict + Z * np.sqrt(Var_error)} , index = Index)
    return df
    
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
    return theta , v_n[:n]#*把完整的迭代的theta作为结果传回去

def NewMessage_predict(X , Gamma , h):
    '''
    输入已经观测到的时间序列,dim=1xn
    Gamma:输入的协方差矩阵 
    h:h步预报
    并返回X1~X_n+h的预报以及X_n+1~X_n+h的均方误差
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
    theta , v_n= NewMessage_parameters(CovMatrix)#* 获得对应的新息系数
    X_h = np.array([])#* 创建h步预报的向量 (总长度为n+h)
    for j in range(n): #*前n个预报
        if(j == 0):
            X_jplus1 = 0 
        else:
            X_jplus1 = np.matmul(theta[j-1] , np.flip(X[:j] - X_h))
        X_h = np.append(X_h , X_jplus1)
    print(len(X_h))
    for i in range(1,h+1):
        X_nplusi = np.matmul(theta[n-2+i][i-1:] , np.flip(X - X_h[:n]))
        X_h = np.append(X_h , X_nplusi)
    #* 下面计算均方误差(返回的数组长度为h)
    MSE = np.array([])
    for i in range(1,h+1):
        MSE = np.append(MSE , Gamma[0][0] - np.matmul(theta[n-2+i][i-1:]**2 , np.flip(v_n[:n])))
    return X_h , MSE

if __name__ == "__main__":
    a = np.array([1,0.1,0.12])
    b = np.array([1,-0.7])
    G = AutoCovariance(a , b , 1 , n=50 , j_max=50)
    X = np.array([0.644 , -0.442 , 0.919 , -1.573 , 0.852 , -0.907 , 0.686 , -0.753 , -0.954 , 0.576])
    #print(Inverseparameters(a,b))
    #print(NewMessage_predict(X , G , 3))
    #print(LinearPredict(X , a , b , 3 , sigma2 = 1))
    #*print(Confidence_Region(X , a , b , 3 , sigma2 = 1))
    # print(Confidence_Region(np.array([101 , 96 , 97.2]) , np.array([1. , 0.6 , 0.3]) , np.array([1]) , 3 ,sigma2 = 36 , mu=10.0))
    # print(LinearPredict(np.array([101 , 96 , 97.2]) , np.array([1. , 0.6 , 0.3]) , np.array([1]) , 3 ,sigma2 = 36 , mu=10.0))
    # print(Inverseparameters(np.array([1. , 0.6 , 0.3]) , np.array([1]) , length = 3))
    c = np.array([1.0])
    d = np.array([1. , -1.1 , 0.28])
    X = np.array([-1.222 , 1.707 , 0.049 , 1.903 , -3.341 , 3.041 , -1.012 , -0.779 , 1.837 , -3.693])
    print(Confidence_Region(X , c , d , 3 , sigma2 = 1))
    print(LinearPredict(X , c , d , 3 , sigma2 = 1))