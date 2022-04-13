import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse , Rectangle
import numpy as np
import pandas as pd
from pytest import TempdirFactory
from scipy.stats import f # F分布相关的api
from scipy.stats import norm # 标准正太的api
from scipy.stats import chi2 # 卡方分布的api
from scipy.stats import t # t分布的api
from math_packages import distance
from math_packages import static_test

def PCA_select(data , use_cov = True , persentage = 0.85 , standard = 0.3):
    '''
    传入数据框data,
    默认使用协方差矩阵进行PCA分解,如果use_cov = False那么就会使用相关系数矩阵进行分解
    传入希望的主成分权重(主成分对应特征根和/总特征根和),默认0.85,需要0<persentage<1
    standard = 共线性性的判断标准,即求解的最小特征值小于0.3
    '''
    if(persentage>1 or persentage<0):
        print('权重需要介于0,1之间,请重新输入权重')
        exit(-1)
    if(use_cov):
        under_resolve_matrix = data.cov()
    else:
        under_resolve_matrix = data.corr(method = 'pearson')
    lambdaarray , vectorarray = np.linalg.eig(under_resolve_matrix) #* 获取这个矩阵的特征值和特征向量 
    if(np.min(lambdaarray)<standard):#* 共线性性的判断标准
        print('最小特征值小于%.2f'%standard)
        exit(-1)
    lambda_reverse = np.sort(lambdaarray)[::-1] ; lambda_reverse_index = np.argsort(lambdaarray)[::-1]
    k = 1
    while(np.sum(lambda_reverse[:k])/np.sum(lambdaarray) < persentage): #*当选取的主成分对应的特征值和占比小于persentage时,增加一个选择
        k += 1
    picked_lambda = lambda_reverse[:k] #* 获得前k个大的主成分向量对应的特征值
    picked_vector = vectorarray[lambda_reverse_index[:k]]

    return picked_lambda , picked_vector #* 返回筛选出来的主成分和对应的特征向量