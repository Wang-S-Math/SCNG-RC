import numpy as np
from numpy import zeros,tanh
from zxgj1.rcnp import rcnp
def rand(*args):
    rng=np.random.default_rng()
    return rng.random(args)
def CCB4quxu(xx,z1=3,z2=0,qt=0):
    l1,l2=xx.shape
    ljxl=[]
    if qt==0:
        for ii in range(l1):
            cc=list(range(z1*ii,z1*ii+z1))
            for jj in range(l2):
                if xx[ii,jj]>0.001:
                    cc.append(jj*z1+z2)
            ljxl.append(cc)
    else:
        for ii in range(l1):
            cc=list(range(z1*ii,z1*ii+z1))
            for jj in range(l2):
                if xx[ii,jj]>0.001:
                    for c in range(jj*z1,jj*z1+z1):
                        cc.append(c)    
            ljxl.append(cc)
    return ljxl

##效果最好为CCB4_2_2_2,CCB4_2_2_1,CCB4_1_2,CCB4_2_1

##数入矩阵 部分+部分 ，输入变量全+部
def CCB4(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin1(nn,n1,1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B[0].fbijiao(yc,y2,0.1)
    return B,yc

##输入矩阵全+全，输入变量 全+全
def CCB41(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,qt=1)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin1(nn,n1,3,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc

## 输入全连接，输入变量全+全
def CCB4_1_1(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,qt=1)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin(nn,n1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc       

##输入矩阵全，输入变量 全+部
def CCB4_1_2(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,3,0,qt=0)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin(nn,n1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc       

## 输入全+全，输入变量全+全
CCB4_2_1=CCB41

## 输入全+部，输入变量全+部
###输入节点全相同
def CCB4_2_2_1(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,3,0,qt=0)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin2(nn,n1,3,1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc       


## 输入全+部，输入变量全+部
###输入节点等比例

def CCB4_2_2_2(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,3,0,qt=0)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin3(nn,n1,3,1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc  

## 输入部+部，输入变量全+全
def CCB4_3_1(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,qt=1)
    Nm=ww.shape[0]
    for zz in range(Nm):
        B1=rcnp()
        ljx=ljxl[zz]
        n1=len(ljx)
        B1.win=B1.fwin1(nn,n1,1,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,pbj)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ljx],grg)
        B.append(B1)
    for ii in range(Nm):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,Nm*3))
    rrduo=zeros((Nm,nn))
    for ii in range(Nm):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(Nm):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(Nm):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B[0].fbijiao(yc,y2,0.1)
    return B,yc


## 输入部+部，输入变量全+部
CCB4_3_2=CCB4
























