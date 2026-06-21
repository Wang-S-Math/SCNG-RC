# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:08:19 2025

@author: lenovo
"""

import numpy as np
from numpy import zeros
from numpy import tanh,eye
from zxgj1.baserc import cbaseesn

rng=np.random.default_rng()


def train(self,xx:np.ndarray,grg=0.7,r0=0,type1=0):
    self.grg=grg
    if np.abs(np.max(self.win)*np.max(xx))>3:
        print("可能未调节win")
    RR=zeros((xx.shape[0],self.wres.shape[0]))
    if isinstance(r0, int):
        r0=fxulex00(self,xx,r0,type1)        
    rr=r0.copy()
    if xx.ndim==1:
        w_in1=np.squeeze (self.win)
        for ii in range(xx.shape[0]):
            rr=(1-grg)*rr+grg*tanh(w_in1*xx[ii]+self.wres@rr+self.bk)
            RR[ii]=rr
    elif xx.ndim==2:
        for ii in range(xx.shape[0]):
            rr=(1-grg)*rr+grg*tanh(self.win@xx[ii]+self.wres@rr+self.bk+self.win1)
            RR[ii]=rr
    else:
        print('维度不对')
    return RR,r0,rr
            
def fxulex00(self,xx,n=0,type1=0):
    if type1==1:
        if n==0:
            n=1000
        r0=rng.random((self.wres.shape[0]))-0.5
        if len(xx)<n:
            x_x=xx-xx[0]
            xx00=xx[0]-x_x
        else:
            xx00=2*xx[0]-xx[:n]
        xx00=xx00[::-1]
        if xx.ndim==1:
            w_in1=np.squeeze (self.win)
            for ii in range(xx00.shape[0]):
                r0=(1-self.grg)*r0+self.grg*tanh(w_in1*xx00+self.wres@r0+self.bk)
        elif xx.ndim==2:
            for ii in range(xx00.shape[0]):
                r0=(1-self.grg)*r0+self.grg*tanh(self.win@xx00+self.wres@r0+self.bk)
        else:
            print('维度不对')
    elif type1==2:
        if n==0:
            n=1000
        r0=rng.random((self.wres.shape[0]))-0.5
        if xx.ndim==1:
            w_in1=np.squeeze (self.win)
            for ii in range(n):
                r0=(1-self.grg)*r0+self.grg*tanh(w_in1*xx[0]+self.wres@r0+self.bk)
        elif xx.ndim==2:
            for ii in range(n):
                r0=(1-self.grg)*r0+self.grg*tanh(self.win@xx[0]+self.wres@r0+self.bk)
        else:
            print('维度不对')  
    else:
        r0=2*n*(rng.random((self.wres.shape[0]))-0.5)
    return r0
def yuce1(self,out,rr,n=2000):
    if out.ndim==1:
        UU=zeros(n)
        w_in1=np.squeeze (self.win)
        for ii in range(n):
            uu=out@rr
            rr=(1-self.grg)*rr+self.grg*tanh(self.wres@rr+w_in1*uu+self.bk)
            UU[ii]=uu
        self.UU=UU
        return UU 
    UU=zeros((n,out.shape[0]))
    for ii in range(n):
        uu=out@rr
        rr=(1-self.grg)*rr+self.grg*tanh(self.wres@rr+self.win@uu+self.bk)
        UU[ii]=uu
    self.UU=UU
    return UU
def canshuganzhi(y1,y2,nn=120,grla=1e-6,pbj=0.95,grg=0.7,grso=1.2,xshl=0.03,she=500,oo1=1,oo2=0.5):
    A=cbaseesn()
    l1,l2,l3=y1.shape
    grsi=grso/np.max(y1)
    A.win=A.fwin(nn, l3,grsi,0)
    A.wres=A.fwres(nn,xshl,pbj)
    A.bk=A.fbk(nn,oo1,oo2)
    pass