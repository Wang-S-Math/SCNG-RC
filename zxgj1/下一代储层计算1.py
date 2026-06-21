# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:43:24 2025

@author: lenovo
"""
import numpy as np
from numpy import zeros,ones_like
from numpy import tanh,eye,sin

class c下一代储层计算():
    def __init__(self, 总项个数=None):
        self.x1=None
        self.x2=None
        self.out=None
        self.RR=None
        self.out1=None
        self.yy=None
        self.wu=None
        self.yc=None
        self.dt=None
        self.start=None
        self.qstart=None
        self.dLt=None
        self.总项个数 = 总项个数  # 添加：支持自定义总项数
    
    def m总程序(self, xx1, dLt=3, dt=1, start=90, qstart=100, grla=1e-5, n=1000):
        RR=self.xunlian(xx1, dLt, dt, start)
        wout=self.m求解输出矩阵(xx1,RR,qstart,grla)
        rrr=RR[-1].copy()
        yc=self.yuce(rrr, wout,n)
        return yc
        
    def xunlian(self, xx1, dLt=3, dt=1, start=90):
        self.x1=xx1
        self.dLt=dLt
        self.start=start
        self.dt=dt
        l1,l2=xx1.shape
        self.l2=l2
        self.线性项个数=l2*dLt
        self.非线性项个数1=int((self.线性项个数+1)*self.线性项个数/2)
        
        # 如果指定了总项数，使用指定的；否则按标准计算
        if self.总项个数 is None:
            self.总项个数=1+self.线性项个数+self.非线性项个数1
        
        self.RR=zeros((xx1.shape[0],self.总项个数))
        
        if dLt==3:
            olin=np.c_[xx1[start:,:],xx1[start-dt:-dt,:],xx1[start-2*dt:-2*dt,:]]
        elif dLt==2:
            olin=np.c_[xx1[start:,:],xx1[start-dt:-dt,:]]
        else:
            print("暂未实现")
        self.RR[start:,:self.线性项个数]=olin
        zz=self.线性项个数
        for ii in range(olin.shape[1]):
            for jj in range(ii,olin.shape[1]):
                self.RR[start:,zz]=olin[:,ii]*olin[:,jj]
                zz=zz+1
        self.RR[start:,zz]=1
        return self.RR
    def m求解输出矩阵(self,xx1,RR,qstart=100,grla=1e-5):
        self.qstart=qstart
        self.grla=grla
        qxx=xx1[qstart+1:]
        qRR=RR[qstart:-1]
        wout=self.qiujie(qxx, qRR,grla)
        return wout
    def qiujie(self,xx,RR,grla=1e-5):
        wout=xx.T@RR@np.linalg.inv(RR.T@RR+grla*eye(RR.shape[1]))
        # wout=xx.T@RR@np.linalg.pinv(RR.T@RR+grla*eye(RR.shape[1]))
        wout1=None
        self.out=wout
        self.out1=wout1
        self.yy=RR@wout.T
        self.wu=xx-self.yy
        return self.out
    def yuce(self,rrr,wout,n=2000):
        yc=zeros((100+n,self.l2))
        yc[:100]=self.x1[-100:]
        xulie=np.triu_indices(self.线性项个数, k=0, m=None)
        for ii in range(100,100+n):
            yc[ii]=wout@rrr
            if (yc[ii]>1e5).any() or (yc[ii]<-1e5).any():
                yc[ii:]=1e5
                break
            if self.dLt==3:
                olin=np.r_['1,2,1',yc[ii],yc[ii-self.dt],yc[ii-2*self.dt]]
            elif self.dLt==2:
                olin=np.r_['1,2,1',yc[ii],yc[ii-self.dt]]
            else:
                print("暂未实现")
            onlin=olin.T@olin
            rr2=onlin[xulie]
            rrr=np.hstack((olin[0],rr2,1))
        self.yc=yc
        return yc
