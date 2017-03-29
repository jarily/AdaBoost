# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:19:42 2017

@author: Jarily
"""
# coding: UTF-8
import numpy as np
from AdaBoost import AdaBoost

def main():
    
    global dataset
    dataset = np.loadtxt('data.txt', delimiter=",")    
    
    global n1
    n1=500
 
    X1 = dataset[0:n1,0:7]  #前n1行的0到7列
    y_tmp = dataset[0:n1,8]    #前n1行的第8列
    y=np.ones(y_tmp.shape)
    y[y_tmp==1]=1
    y[y_tmp==0]=-1
    y1=y;
    #print("y==")
    #print(y1)
    X=[0,1,2,3,4,5,6,7,8,9]
    y=[1,1,1,-1,-1,-1,1,1,1,-1]
    
    X1=X1.transpose()
    #X1=np.array([X])
    #y1=np.array(y)
    
    X2 =  dataset[n1:,0:7] 
    y_tmp =  dataset[n1:,8] 
    
    y=np.ones(y_tmp.shape)
    y[y_tmp==1]=1
    y[y_tmp==0]=-1
    
    y2=y;
    
    X2=X2.transpose()
    #print(X1)
    
    ada=AdaBoost(X1,y1)
    #print("hehe1")
    ada.train(10)
    #print("hehe2")
   # print(ada.finalclassifer(3))
    #print("res==")
    #res=ada.pred([[111.55,1.1,5.35],[4.4,2.8,0.9]])
    res=ada.pred(X2)
    res=res.tolist() # 转换成list
    y2=y2.tolist()
    #print(type(y2))
    #print(type(res))
    
    cnt=0
    sum=0
    for i in range (len(res)):
        if y2[i]==res[i]:
            cnt+=1
        sum+=1
    print("测试样本总数：%d"%sum)
    print("测试正确样本数：%d"%cnt)
    print("正确率为：%.2lf"%(1.0*cnt/sum))

if __name__=='__main__':
    main()