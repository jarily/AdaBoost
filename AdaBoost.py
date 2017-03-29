# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:19:42 2017

@author: Jarily
"""
# coding: UTF-8

import numpy as np
from WeakClassify import DecisionStump


class AdaBoost:
    def __init__(self,X,y,Weaker=DecisionStump):
        self.X=np.array(X)
        self.y=np.array(y).flatten(1)
        self.Weaker=Weaker
        self.sums=np.zeros(self.y.shape)
        
        '''
        W为权值，初试情况为均匀分布，即所有样本都为1/n
        '''
        self.W=np.ones((self.X.shape[1],1)).flatten(1)/self.X.shape[1]

        self.Q=0  #弱分类器的实际个数
        
       # M 为弱分类器的最大数量，可以在main函数中修改
        
    def train(self,M=5):
        self.G={}         # 表示弱分类器的字典
        self.alpha={}     # 每个弱分类器的参数
        for i in range(M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(M):   # self.G[i]为第i个弱分类器
            self.G[i]=self.Weaker(self.X,self.y)
            e=self.G[i].train(self.W) #根据当前权值进行该个弱分类器训练
            self.alpha[i]=1/2*np.log((1-e)/e) #计算该分类器的系数
            res=self.G[i].pred(self.X)  #res表示该分类器得出的输出
            # Z表示规范化因子
            Z=self.W*np.exp(-self.alpha[i]*self.y*res.transpose())
            self.W=(Z/Z.sum()).flatten(1) #更新权值
            self.Q=i
            # errorcnt返回分错的点的数量，为0则表示perfect
            if (self.errorcnt(i)==0):
                print("%d个弱分类器可以将错误率降到0"%(i+1))
                break
            

    def errorcnt(self,t):   #返回错误分类的点
        self.sums=self.sums+self.G[t].pred(self.X).flatten(1)*self.alpha[t]
        
        pre_y=np.zeros(np.array(self.sums).shape)
        pre_y[self.sums>=0]=1
        pre_y[self.sums<0]=-1
        
        t=(pre_y!=self.y).sum() 
        return t
    
    def pred(self,test_X):  #测试最终的分类器
        test_X=np.array(test_X)
        sums=np.zeros(test_X.shape[1])
        for i in range(self.Q+1):
            sums=sums+self.G[i].pred(test_X).flatten(1)*self.alpha[i]
        pre_y=np.zeros(np.array(sums).shape)
        pre_y[sums>=0]=1
        pre_y[sums<0]=-1
        return pre_y
