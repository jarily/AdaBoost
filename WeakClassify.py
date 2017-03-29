# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:19:42 2017

@author: Jarily
"""

import numpy as np

'''
Decision Stump  单层决策树算法   弱分类器
'''

class DecisionStump:
    def __init__(self,X,y):
        self.X=np.array(X)
        self.y=np.array(y)
        self.N=self.X.shape[0]
        

    
    def train(self,W,steps=100):  #返回所有参数中阈值最小的
        '''    
        W长度为N的向量,表示N个样本的权值  
        threshold_value为阈值
        threshold_pos为第几个参数
        threshold_tag为1或者-1.大于阈值则分为threshold_tag，小于阈值则相反
        '''
        min = float("inf")    #将min初始化为无穷大
        threshold_value=0;
        threshold_pos=0;
        threshold_tag=0;
        self.W=np.array(W)
        for i in range(self.N):  #  value表示阈值，errcnt表示错误的数量
            value,errcnt = self.findmin(i,1,steps)
            if (errcnt < min):
                min = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = 1
        for i in range(self.N):  # -1
            value,errcnt= self.findmin(i,-1,steps)
            if (errcnt < min):
                min = errcnt
                threshold_value = value
                threshold_pos = i
                threshold_tag = -1
        #最终更新
        self.threshold_value=threshold_value
        self.threshold_pos=threshold_pos
        self.threshold_res=threshold_tag
        print(self.threshold_value,self.threshold_pos,self.threshold_res)
        return min
    
    def findmin(self,i,tag,steps):  #找出第i个参数的最小的阈值,tag为1或-1
        t = 0
        tmp = self.predintrain(self.X,i,t,tag).transpose()
        errcnt = np.sum((tmp!=self.y)*self.W)
        #print now
        buttom=np.min(self.X[i,:])  #该项属性的最小值，下界
        up=np.max(self.X[i,:])      #该项属性的最大值，上界
        minerr = float("inf")       #将minerr初始化为无穷大
        value=0                     #value表示阈值
        st=(up-buttom)/steps        #间隔
        for t in np.arange(buttom,up,st):
            tmp = self.predintrain(self.X,i,t,tag).transpose()
            errcnt = np.sum((tmp!=self.y)*self.W)
            if errcnt < minerr:
                minerr=errcnt
                value=t
        return value,minerr
    
    def predintrain(self,test_set,i,t,tag): #训练时按照阈值为t时预测结果
        test_set=np.array(test_set).reshape(self.N,-1)
        pre_y = np.ones((np.array(test_set).shape[1],1))
        pre_y[test_set[i,:]*tag<t*tag]=-1
        return pre_y

    def pred(self,test_X):  #弱分类器的预测
        test_X=np.array(test_X).reshape(self.N,-1) #转换为N行X列，-1懒得算
        pre_y = np.ones((np.array(test_X).shape[1],1))
        pre_y[test_X[self.threshold_pos,:]*self.threshold_res<self.threshold_value*self.threshold_res]=-1
        return pre_y
	