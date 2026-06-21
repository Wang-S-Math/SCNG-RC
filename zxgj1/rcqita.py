# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 16:24:21 2025

@author: lenovo
"""
import numpy as np
from numpy import zeros,tanh
from zxgj1.rcnp import rcnp
def CCB5(y1,y2,nn=120,grla=1e-6,pbj=0.95,grg=0.7,
         grso=1.2,xshl=0.03,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    l1,l2=y1.shape
    nm=int(l2//3)
    grsi=grso/np.max(np.abs(y1))
    RR=zeros((l1,nn*nm))
    for ii in range(nm):
        BB1=rcnp()
        BB1.win=BB1.fwin(nn,3,grsi)
        if nn*xshl<1.5:
            xshl=3/nn
        BB1.setwres(nn,xshl,pbj)
        BB1.bk=BB1.fbk(nn,oo1,oo2)
        RR[:,nn*ii:nn*(ii+1)]=BB1.train(y1[:,ii*3:(ii+1)*3],grg)
        B.append(BB1)
    wout=BB1.getwout(y1[she:], RR[she-1:-1],grla)
    rr=RR[-1,:].copy()
    yc=zeros((5000,l2))
    for jj in range(5000):
        yc[jj]=wout@rr
        for ii in range(nm):
            rr[nn*ii:nn*(ii+1)]=B[ii].frun(yc[jj,3*ii:3*(ii+1)],rr[nn*ii:nn*(ii+1)])
    BB1.fbijiao(yc,y2,0.1)
    return B,yc
    
def CCB51(BB,y1,y2,grla=1e-6,she=500,ycn=5000):
    l1,l2=y1.shape
    nm=int(l2//3) 
    nn=BB[0].wres.shape[0]
    RR=zeros((l1,nn*nm))
    for ii in range(nm):
        RR[:,nn*ii:nn*(ii+1)]=BB[ii].RR
    
    wout=BB[0].getwout(y1[she:], RR[she-1:-1],grla)
    rr=RR[-1,:].copy()
    yc=zeros((5000,l2))
    for jj in range(5000):
        yc[jj]=wout@rr
        for ii in range(nm):
            rr[nn*ii:nn*(ii+1)]=BB[ii].frun(yc[jj,3*ii:3*(ii+1)],rr[nn*ii:nn*(ii+1)])
    BB[0].fbijiao(yc,y2,0.1)
    return BB,yc
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
