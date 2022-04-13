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

#*----------------------------------------------------------------
mpl.rcParams['font.sans-serif'] = ['SimHei'] # *允许显示中文
plt.rcParams['axes.unicode_minus']=False# *允许显示坐标轴负数
#*----------------------------------------------------------------

params = {'legend.fontsize': 7,}

plt.rcParams.update(params)

def norm_QQ(X , axes , standardized = True):
    '''
    绘制标准正态QQ图
    传进来一个一维数组,和要绘制的axes(plt的subplot,以及是否要正则化的bool值)
    '''
    temp = X.values
    if(temp.ndim != 1):
        print('数组并非一维,请选用多元正太QQ图')
        exit(-1)
    name = X.name #* 获取这个series的name
    norm_plist = norm.ppf((np.array(range(1,len(temp)+1))-0.5)/len(temp) , 0 , 1) #* 构造标准正太的分位数list
    if(standardized):
        temp = (temp - np.mean(temp))/np.sqrt(np.var(temp)) #*对X进行标准化，因为如果X~N(a,b),则X-a/sqrt(b) ~ N(0 , 1)
        inf = np.min([temp,norm_plist]) ; sup = np.max([temp,norm_plist])
        x = np.arange(inf , sup + 0.1 , 0.1)
        axes.plot(x , x , color = 'green') #* 绘制y=x标准线
    axes.scatter(norm_plist , np.sort(temp) , s = 9 , alpha = 0.6)
    axes.set_title(name)

def chi2_QQ(X , axes):
    '''
    绘制卡方QQ图(对应维数)
    传进来一个n , p维矩阵 , (不能是数据框)
    和要绘制的axes(plt的subplot,以及是否要正则化的bool值)
    '''
    temp = X
    if(X.ndim == 1):
        print('至少为2维才能绘制卡方图')
        exit(-1)
    n , p = X.shape
    chi2_plist = chi2.ppf((np.array(range(1,len(temp)+1))-0.5)/len(temp) , p)
    d = distance.Mahalanobis_Distance(X)
    inf = np.min([d,chi2_plist]) ; sup = np.max([d,chi2_plist])
    x = np.arange(inf , sup + 0.1 , 0.1)
    axes.plot(x , x , color = 'green') #* 绘制y=x标准线
    axes.scatter(chi2_plist , np.sort(d) , s = 9 , alpha = 0.6)
    
def Ellipse_CR(df , axes , scatter = True , rectangle = True , alpha = 0.05):
    '''
    给予一个dataframe,nx2的dataframe,
    构造对应的椭圆置信区间,并在axes中表示
    默认开启散点图绘制,默认开启椭圆边界显示(T2置信区间)
    并且alpha默认为0.05
    '''
    X = df.values ; n,p = X.shape
    S = np.cov(X.T) #* 计算协方差矩阵
    mu = np.average(X , axis=0) #*计算均值向量
    ColumnName = df.columns.tolist() #* 获取列名
    
    F_alpha = f.isf(alpha , p , n-p) #* 显著性水平alpha = 0.05
    params = p*(n-1)/(n-p) * F_alpha  #* 计算对应的准换的参数
    
    lambda_1 , e_1 = np.linalg.eig(S)
    l1 = 2 * np.sqrt(lambda_1[0]) * np.sqrt(params/n) * np.sqrt(sum(e_1[0]**2))
    l2 = 2 * np.sqrt(lambda_1[1]) * np.sqrt(params/n) * np.sqrt(sum(e_1[1]**2))
    angle = np.arctan(e_1[0][1]/e_1[0][0])/(2*np.pi) * 360 #* 转换成角度制
    
    axes.scatter(mu[0] , mu[1] , alpha = 0.8 , color = 'red')
    ellipse = Ellipse(xy = tuple(mu) , width = l1 , height = l2 , angle = angle) #*创建椭圆对象
    axes.add_patch(p=ellipse)
    ellipse.set(alpha=0.4,color='lightskyblue')
    axes.set_xlabel(ColumnName[0]) ; axes.set_ylabel(ColumnName[1])
    
    if(scatter): #*如果散点图参数为True
        axes.scatter(X.T[0] , X.T[1] , s = 9 , alpha = 0.6 , color = '#d58794')
    if(rectangle):
        region = static_test.T2_region(df , alpha = alpha).values #*计算对应的T2置信区间，对象为np.array,方便等会添加矩形
        axes.add_patch(Rectangle(
                xy = (region[0][0] , region[1][0]) , 
                width = region[0][1] - region[0][0] , 
                height = region[1][1] - region[1][0] , fill = False)
            )
    
def test():
    print('没毛病')
    
if __name__ == '__main__':
    test()
    temp = np.array([
            [1.3,2.2],
            [1.,2.],
            [3.,0.5],
            [2.,3.],
            [1.2,2.2],
            [2.3,3.3],
            [3.4,1.3]
    ])
    df=pd.DataFrame(temp , columns = ['x1' , 'x2'])
    fig , axes = plt.subplots(figsize=(7,7) , dpi = 120)
    Ellipse_CR(df , axes)
    plt.show()