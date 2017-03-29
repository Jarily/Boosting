# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:33:35 2017

@author: Jarily
"""

import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier


'''  弱分类器c1 '''
def weak_classifier_1(X,y):    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

'''  弱分类器c2 '''
def weak_classifier_2(X,y):    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

'''  弱分类器c3 '''
def weak_classifier_3(X,y):    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model
   
'''  boosting所获得的强分类器 '''    
def strong_classifier(c1,c2,c3,X):
    y1=c1.predict(X)
    y2=c2.predict(X)
    y3=c3.predict(X)
    
    if y1==y2:
        return y1
    else:
        return y3
    
'''  boosting算法 '''    
def train():
    global dataset
    dataset = np.loadtxt('data.txt', delimiter=",")    
    #print(dataset.shape)


    X1 = dataset[0:n1,0:7]  #前n1行的0到7列
    y1 = dataset[0:n1,8]    #前n1行的第8列
    
    global c1 
    c1 = weak_classifier_1(X1,y1)  #弱分类器c1  

    for i in range(0,n1):
        dataset=np.delete(dataset,[0],axis=0)
   # print(dataset.shape)

    ls=[]  #保存算法应该放入集合D2的点
    flag=0
    for i in range(0,dataset.shape[0]-n2):
        random.seed()              #默认以为系统时间为种子，体现随机性
        xx=random.randint(0,1)     #抛硬币 1为上 0为下
        X1 = dataset[i:i+1,0:7]    #所有行的0到7列
        y1 = dataset[i:i+1,8]      #所有行的第8列
        test_y = c1.predict(X1)    #将当前点放入弱分类器c1种测试
        if xx==1:
            if y1==test_y:         #测试正确
                pass
            else:
                if(flag==1):  #将X1,y1加入X2,y2
                    X2=np.vstack((X2,X1)) 
                    y2=np.vstack((y2,y1))
                else:
                    X2=X1
                    y2=y1
                    flag=1
                ls.append(i)
        else:
            if y1==test_y:
                if(flag==1):
                    X2=np.vstack((X2,X1))
                    y2=np.vstack((y2,y1))
                else:
                    X2=X1
                    y2=y1
                    flag=1
                ls.append(i)
            else:
                pass
            
    global c2
    c2 = weak_classifier_2(X2,y2)  #弱分类器c2  

    
    flag=0
    for i in range(0,dataset.shape[0]-n2):
        if i not in ls:                       #没有放入集合D2的剩下的点 
            X1 = dataset[i:i+1,0:7]  #第i行的0到7列
            y1 = dataset[i:i+1,8]    #第i行的第8列
            test_y1 = c1.predict(X1)
            test_y2 = c2.predict(X1)
            if test_y1 == test_y2:
                pass
            else:
                #print(test_y1)
                #print(test_y2)
                #print("----------")
                if(flag==1):
                    X3=np.vstack((X3,X1))
                    y3=np.vstack((y3,y1))
                else:
                    X3=X1
                    y3=y1
                    flag=1
    
    global c3 
    c3 = weak_classifier_3(X3,y3)  #弱分类器c3  
    

def main():
    
    global n1,n2
    n1=300   # n1为集合D1的点的数量
    n2=50   # n2为测试的点的数量
    
    train()  # boosting算法
    
    cnt=0    #正确的测试
    sum=0    #测试点的总数
    for i in range(dataset.shape[0]-n2,dataset.shape[0]):
        X1 = dataset[i:i+1,0:7]  #第i行的0到7列
        y1 = dataset[i:i+1,8]    #第i行的第8列
        sum+=1
        test_y=strong_classifier(c1,c2,c3,X1)
        if y1== test_y:  #测试正确
            #print("1")
            cnt+=1
        else:
            pass
            #print("0")
    print("测试样本总数：%d"%sum)
    print("测试正确样本数：%d"%cnt)
    print("正确率为：%.2lf"%(1.0*cnt/sum))

if __name__=='__main__':
    main()