import numpy as np
import pandas as pd
from scipy.stats import f # F分布相关的api
from scipy.stats import norm # 标准正太的api
from scipy.stats import chi2 # 卡方分布的api
from scipy.stats import t # t分布的api


from . import distance

def T_2_test(X , mu_0):
    '''
    返回X对应mu_0的T方统计量的值
    '''
    n , m = X.shape
    S = np.cov(X.T) #*求出样本协方差矩阵
    bar_x = np.average(X , axis = 0)
    T_2 = n * np.matmul( np.matmul(bar_x - mu_0 , np.linalg.inv(S)) , (bar_x - mu_0).T)
    return T_2 #* 返回对应的T^2统计量

def Pang_region(df , alpha = 0.05):
    '''
    df为需要构造庞弗罗尼置信区间的原始数据 nxp的数据框,p维待估计参数
    alpha为显著水平,默认为0.05
    返回一个pd.dataframe形式的数据框,记录p维参数的对应置信下上限
    '''
    X = df.values
    n , p = X.shape ; S = np.cov(X.T) ; mu = np.average(X , axis = 0)
    params= t.isf(alpha/(2*p) , n-1)#* 显著性水平alpha 右侧分位数 默认m = p
    inf_pang = [] ; sup_pang = []
    a = np.identity(p)
    for i in range(p):
        inf_pang.append(np.matmul(a[i] , mu.T) - params * np.sqrt(np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
        sup_pang.append(np.matmul(a[i] , mu.T) + params * np.sqrt(np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
    ColNames_List = df.columns.tolist() #* 获取数据框的列名
    region = pd.DataFrame({'置信下限':inf_pang , '置信上限': sup_pang} , index=ColNames_List)
    return region

def T2_region(df , alpha = 0.05):
    '''
    df为需要构造T2置信区间的原始数据 nxp的数据框,p维待估计参数
    alpha为显著水平,默认为0.05
    返回一个pd.dataframe形式的数据框,记录p维参数的对应置信下上限
    '''
    X = df.values
    n , p = X.shape ; S = np.cov(X.T) ; mu = np.average(X , axis = 0)
    F_alpha = f.isf(alpha , p , n-p) #* 显著性水平alpha = 0.05
    params_1 = p*(n-1)/(n-p) * F_alpha
    inf_t2 = [] ; sup_t2 = []
    a = np.identity(p)
    for i in range(p):
        inf_t2.append(np.matmul(a[i] , mu.T) - np.sqrt(params_1 * np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
        sup_t2.append( np.matmul(a[i] , mu.T) + np.sqrt(params_1 * np.matmul(np.matmul(a[i] , S) , a[i].T) / n))
    ColNames_List = df.columns.tolist() #* 获取数据框的列名
    region = pd.DataFrame({'置信下限':inf_t2 , '置信上限': sup_t2} , index=ColNames_List)
    return region