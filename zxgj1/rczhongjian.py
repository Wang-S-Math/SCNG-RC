# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:19:21 2025

@author: lenovo
"""

from zxgj1.baserc import cbaseesn
import numpy as np
from numpy import zeros,tanh,eye

def f维度转换(xx:np.ndarray,wei1:int=0):
    if xx.ndim==1:
        print('维度为1')
    elif xx.ndim==2:
        l1,l2=xx.shape
        if wei1==0:
            print('未输入wei')
            return
        l3=l2//wei1
        yy=zeros((l1,wei1,l3))
        for ii in range(l3):
            yy[:,:,ii]=xx[:,ii*wei1:ii*wei1+wei1]
        yy=yy.transpose(2,0,1)
        return yy
    elif xx.ndim==3:
        l1,l2,l3=xx.shape
        yy=zeros((l2,l1*l3))
        for ii in range(l1):
            yy[:,ii*l3:ii*l3+l3]=xx[ii,:,:]
        return yy
    else:
        print(f'维度为{xx.ndim},不正确')
        return 0
def CCB5(y1,y2,ww,nn=120,ouhe=40,grla=1e-6,grg=0.7,xshl=0.03,she=500,
ycn=5000,grgo=1.2,oo1=1,oo2=0.5,pbj=0.95):
    ##    储层耦合f(+（-）)
    rng=np.random.default_rng()
    y1=f维度转换(y1,3)
    y2=f维度转换(y2,3)
    grgs=grgo/np.max(y1)
    l1,l2,l3=y1.shape
    B=[]
    for ii in range(l1):   
        A=cbaseesn()
        A.win=A.fwin(nn,l3,grgs)
        A.wres=A.fwres(nn,xshl,pbj)
        A.bk=A.fbk(nn,oo1,oo2)
        B.append(A)
    r0=rng.random((l1,nn))
    RR=rng.random((l1,l2,nn))
    rr=r0
    r1=rr.copy()
    for ii in range(l2):
        for jj in range(l1):
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@y1[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk+0.1*(ww[jj]@rr-rr[jj]))
        RR[:,ii,:]=r1
        rr=r1.copy()
    wout=[]
    RRR=RR[:,she:-1,:]
    quy1=y1[:,she+1:,:]
    for ii in range(l1):
        wout1=quy1[ii].T@RRR[ii]@np.linalg.pinv(RRR[ii].T@RRR[ii]+grla*eye(RRR[ii].shape[1]))
        wout.append(wout1)
    yc=zeros((l1,5000,3))
    for ii in range(5000):
        for jj in range(l1):
            yc[jj,ii]=wout[jj]@rr[jj]
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@yc[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk+0.1*(ww[jj]@rr-rr[jj]))
            rr=r1.copy()
    B[0].fbijiao(f维度转换(yc),f维度转换(y2),0.1)
    return B,f维度转换(yc)

def CCB51(y1,y2,ww,nn=120,ouhe=40,grla=1e-6,grg=0.7,xshl=0.03,she=500,
ycn=5000,grgo=1.2,oo1=1,oo2=0.5,pbj=0.95):
    ##    储层耦合f()+（-）
    rng=np.random.default_rng()
    y1=f维度转换(y1,3)
    y2=f维度转换(y2,3)
    grgs=grgo/np.max(y1)
    l1,l2,l3=y1.shape
    B=[]
    for ii in range(l1):   
        A=cbaseesn()
        A.win=A.fwin(nn,l3,grgs)
        A.wres=A.fwres(nn,xshl,pbj)
        A.bk=A.fbk(nn,oo1,oo2)
        B.append(A)
    r0=rng.random((l1,nn))
    RR=rng.random((l1,l2,nn))
    rr=r0
    r1=rr.copy()
            
    for ii in range(l2):
        for jj in range(l1):
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@y1[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk)+0.1*(ww[jj]@rr-rr[jj])
        RR[:,ii,:]=r1
        rr=r1.copy()
    wout=[]
    RRR=RR[:,she:-1,:]
    quy1=y1[:,she+1:,:]
    for ii in range(l1):
        wout1=quy1[ii].T@RRR[ii]@np.linalg.pinv(RRR[ii].T@RRR[ii]+grla*eye(RRR[ii].shape[1]))
        wout.append(wout1)
    yc=zeros((l1,5000,3))
    for ii in range(5000):
        for jj in range(l1):
            yc[jj,ii]=wout[jj]@rr[jj]
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@yc[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk)+0.1*(ww[jj]@rr-rr[jj])
            rr=r1.copy()
    B[0].fbijiao(f维度转换(yc),f维度转换(y2),0.1)
    
    return B,f维度转换(yc)

def CCB52(y1,y2,ww,nn=120,ouhe=40,grla=1e-6,grg=0.7,xshl=0.03,she=500,
ycn=5000,grgo=1.2,oo1=1,oo2=0.5,pbj=0.95):
    ##    储层耦合f(+)
    rng=np.random.default_rng()
    y1=f维度转换(y1,3)
    y2=f维度转换(y2,3)
    grgs=grgo/np.max(y1)
    l1,l2,l3=y1.shape
    B=[]
    for ii in range(l1):   
        A=cbaseesn()
        A.win=A.fwin(nn,l3,grgs)
        A.wres=A.fwres(nn,xshl,pbj)
        A.bk=A.fbk(nn,oo1,oo2)
        B.append(A)
    r0=rng.random((l1,nn))
    RR=rng.random((l1,l2,nn))
    rr=r0
    r1=rr.copy()
    for ii in range(l2):
        for jj in range(l1):
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@y1[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk+0.1*ww[jj]@rr)
        RR[:,ii,:]=r1
        rr=r1.copy()
    wout=[]
    RRR=RR[:,she:-1,:]
    quy1=y1[:,she+1:,:]
    for ii in range(l1):
        wout1=quy1[ii].T@RRR[ii]@np.linalg.pinv(RRR[ii].T@RRR[ii]+grla*eye(RRR[ii].shape[1]))
        wout.append(wout1)
    yc=zeros((l1,5000,3))
    for ii in range(5000):
        for jj in range(l1):
            yc[jj,ii]=wout[jj]@rr[jj]
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@yc[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk+0.1*ww[jj]@rr)
            rr=r1.copy()
    B[0].fbijiao(f维度转换(yc),f维度转换(y2),0.1)
    return B,f维度转换(yc)

def CCB53(y1,y2,ww,nn=120,ouhe=40,grla=1e-6,grg=0.7,xshl=0.03,she=500,
ycn=5000,grgo=1.2,oo1=1,oo2=0.5,pbj=0.95):
    ##    储层耦合f()+
    rng=np.random.default_rng()
    y1=f维度转换(y1,3)
    y2=f维度转换(y2,3)
    grgs=grgo/np.max(y1)
    l1,l2,l3=y1.shape
    B=[]
    for ii in range(l1):   
        A=cbaseesn()
        A.win=A.fwin(nn,l3,grgs)
        A.wres=A.fwres(nn,xshl,pbj)
        A.bk=A.fbk(nn,oo1,oo2)
        B.append(A)
    r0=rng.random((l1,nn))
    RR=rng.random((l1,l2,nn))
    rr=r0
    r1=rr.copy()
            
    for ii in range(l2):
        for jj in range(l1):
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@y1[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk)+0.1*ww[jj]@rr
        RR[:,ii,:]=r1
        rr=r1.copy()
    wout=[]
    RRR=RR[:,she:-1,:]
    quy1=y1[:,she+1:,:]
    for ii in range(l1):
        wout1=quy1[ii].T@RRR[ii]@np.linalg.pinv(RRR[ii].T@RRR[ii]+grla*eye(RRR[ii].shape[1]))
        wout.append(wout1)
    yc=zeros((l1,5000,3))
    for ii in range(5000):
        for jj in range(l1):
            yc[jj,ii]=wout[jj]@rr[jj]
            r1[jj]=(1-grg)*rr[jj]+grg*tanh(B[jj].win@yc[jj,ii]+B[jj].wres@rr[jj]+B[jj].bk)+0.1*ww[jj]@rr
            rr=r1.copy()
    B[0].fbijiao(f维度转换(yc),f维度转换(y2),0.1)
    
    return B,f维度转换(yc)


























