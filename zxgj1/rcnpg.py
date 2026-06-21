import numpy as np
from numpy import zeros,tanh
from zxgj1.rcnp import rcnp
from zxgj1.validpt import Vpt
def rand(*args):
    rng=np.random.default_rng()
    return rng.random(args)
def shijian(xx,yy):
    return np.array([
    Vpt.vpt1(xx, yy, 0.1),
    Vpt.vpt1(xx, yy, 0.2),
    Vpt.vpt1(xx, yy, 0.3),
    Vpt.vpt1(xx, yy, 0.4),
    Vpt.vpt1(xx, yy, 0.5),
    Vpt.vpt1(xx, yy, 0.6),
    Vpt.vpt1(xx, yy, 0.7),
    Vpt.vpt2(xx, yy, 0.15),
    Vpt.vpt3(xx, yy, 0.4),
    Vpt.vpt4(xx, yy, 0.4),
    Vpt.vpt5(xx, yy, 0.4),
    Vpt.vpt3(xx, yy, 0.5),
    Vpt.vpt3(xx, yy, 0.8),])
def CCB1(y1,y2,nn=120,grla=1e-6,pbj=0.95,grg=0.7,grso=1.2,xshl=0.03,she=500,oo1=1,oo2=0.5):
    l1,l2=y1.shape
    B=rcnp()
    grgs=grso/np.max(y1)
    B.win=B.fwin(nn,l2,grgs)
    if nn*xshl<1.5:
        xshl=3/nn
    B.setwres(nn,xshl,pbj)
    B.bk=B.fbk(nn,oo1,oo2)
    B.train(y1,grg)
    B.getwoutshe(y1,she,grla)
    B.yuce1(B.rr,5000)
    B.fbijiao(B.UU,y2,0.1)
    return B,B.UU
def CCB11(y1,y2,nn=120,grla=1e-6,grg=0.7,pbj=0.95,grso=1.2,xshl=0.03,she=500,oo1=1,oo2=0.5):
    l1,l2=y1.shape
    B=rcnp()
    grgs=grso/np.max(y1)
    B.win=B.fwin1(nn,l2,3,grgs)
    if nn*xshl<1.5:
        xshl=3/nn
    B.setwres(nn,xshl,pbj)
    B.bk=B.fbk(nn,oo1,oo2)
    B.train(y1,grg)
    B.getwoutshe(y1,she,grla)
    B.yuce1(B.rr,5000)
    B.fbijiao(B.UU,y2,0.1)
    return B,B.UU
def CCB21(y1,y2,nn=120,grla=1e-6,grg=0.7,pbj=1.05,grso=1.2,xshl=0.03,she=500,oo1=1,oo2=0.5):
    l1,l2=y1.shape
    B=rcnp()
    grgs=grso/np.max(y1)
    B.setwin(nn,l2,grgs)
    if nn*xshl<1.5:
        xshl=3/nn
    B.setwres(nn,xshl,pbj)
    B.bk=B.fbk(nn,oo1,oo2)
    B.train(y1,grg)
    B.getwoutshe(y1,she,grla)
    B.yuce1(B.rr,5000)
    # B.bijiao(y2,B.UU)
    B.fbijiao(B.UU,y2,0.1)
    return B
def CCB2(y1,y2,nn,grla=1e-6,grg=0.7,oo1=1,oo2=0.5):
    BA=[]
    l1,l2=y1.shape
    yc2=zeros((5000,l2))
    for ii in range(int(l2/3)):
        BA1=CCB21(y1[:,3*ii:3*ii+3],y2[:,3*ii:3*ii+3],nn,grla,grg,oo1=oo1,oo2=oo2)
        yc2[:,3*ii:3*ii+3]=BA1.UU
        BA.append(BA1)
    BA1.fbijiao(yc2,y2,0.1)
    return BA,yc2
def qiufei0(xx1):
    zz=[]
    for ii in range(xx1.shape[0]):
        zzz=[]
        for jj in range(xx1.shape[1]):
            if xx1[ii,jj]>0.001:
                zzz.append(jj)
        zz.append(zzz)    
    return(zz)

def CCB3(y1,y2,ww,nn=120,ouhe=40,grla=1e-6,grg=0.7,xshl=0.03,she=500,ycn=5000,oo1=1,oo2=0.5):
    BB=[]
    grgs=1.2/np.max(y1)
    l2=y1.shape[1]
    mn=int(l2/3)
    for ii in range(mn):
        B1=rcnp()
        B1.setwin(nn,3,grgs)
        if nn*xshl<1.5:
            xshl=3/nn
        B1.setwres(nn,xshl,0.95)
        B1.bk=B1.fbk(nn,oo1,oo2)
        B1.train(y1[:,ii*3:(ii+1)*3],grg)
        BB.append(B1)
    print('储层生成')
    RRq=[]
    wl=qiufei0(ww)
    for ii in range(mn):
        RRqh=BB[ii].RR
        for jj in wl[ii]:
                RRqh=np.hstack((RRqh,BB[jj].RR[:,:ouhe]))
        RRq.append(RRqh)
    for ii in range(mn):
        BB[ii].getwout(y1[she:,ii*3:(ii+1)*3],RRq[ii][she-1:-1],grla)
    print('已求解wout')
  
    yc=zeros((ycn,l2))
    rrduo=zeros((mn,nn))
    for ii in range(mn):
        rrduo[ii]=BB[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(mn):
            rrh=rrduo[ii]
            for jj in wl[ii]:
                rrh=np.hstack((rrh,rrduo[jj][:ouhe]))
            yc[zz,ii*3:(ii+1)*3]=BB[ii].out@rrh
        for ii in range(mn):
            rrduo[ii]=(1-BB[ii].grg)*rrduo[ii]+BB[ii].grg*tanh(BB[ii].win@yc[zz,ii*3:(ii+1)*3]+BB[ii].wres@rrduo[ii]+BB[ii].bk)
    print('预测完成')
    BB[0].fbijiao(yc,y2,0.1)
    return BB,yc
import copy
def CCB31(BB,y1,y2,ww,ouhe=40,grla=1e-7,she=500,ycn=5000):
    CBA = copy.deepcopy(BB)
    BB=CBA
    nn=BB[1].wres.shape[0]
    l2=y1.shape[1]
    mn=int(l2/3)
    RRq=[]
    wl=qiufei0(ww)
    for ii in range(mn):
        RRqh=BB[ii].RR
        for jj in wl[ii]:
                RRqh=np.hstack((RRqh,BB[jj].RR[:,:ouhe]))
        RRq.append(RRqh)
    for ii in range(mn):
        BB[ii].getwout(y1[she:,ii*3:(ii+1)*3],RRq[ii][she-1:-1],grla)
    print('已求解wout')
  
    yc=zeros((ycn,l2))
    rrduo=zeros((mn,nn))
    for ii in range(mn):
        rrduo[ii]=BB[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(mn):
            rrh=rrduo[ii]
            for jj in wl[ii]:
                rrh=np.hstack((rrh,rrduo[jj][:ouhe]))
            yc[zz,ii*3:(ii+1)*3]=BB[ii].out@rrh
        for ii in range(mn):
            rrduo[ii]=(1-BB[ii].grg)*rrduo[ii]+BB[ii].grg*tanh(BB[ii].win@yc[zz,ii*3:(ii+1)*3]+BB[ii].wres@rrduo[ii]+BB[ii].bk)
    print('预测完成')
    BB[0].fbijiao(yc,y2)
    return BB,yc
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
def CCB4_2_2_2(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,3,0,qt=0)
    for zz in range(7):
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
    for ii in range(7):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,21))
    rrduo=zeros((7,nn))
    for ii in range(7):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(7):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(7):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc  

def CCB4(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww)
    for zz in range(7):
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
    for ii in range(7):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,21))
    rrduo=zeros((7,nn))
    for ii in range(7):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(7):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(7):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B[0].fbijiao(yc,y2,0.1)
    return B,yc

    
def CCB41(y1,y2,ww,nn=120,grla=1e-6,grg=0.7,pbj=0.95,xshl=0.03,grso=1.2,she=500,ycn=5000,oo1=1,oo2=0.5):
    B=[]
    grgs=grso/np.max(y1)
    ljxl=CCB4quxu(ww,qt=1)
    for zz in range(7):
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
    for ii in range(7):
        B[ii].getwoutshe(y1[:,ii*3:ii*3+3],she,grla)
    yc=zeros((ycn,21))
    rrduo=zeros((7,nn))
    for ii in range(7):
        rrduo[ii]=B[ii].rr.copy()
    for zz in range(ycn):
        for ii in range(7):
            yc[zz,ii*3:(ii+1)*3]=B[ii].out@rrduo[ii]
        for ii in range(7):
            rrduo[ii]=(1-B[ii].grg)*rrduo[ii]+B[ii].grg*tanh(B[ii].win@yc[zz,ljxl[ii]]+B[ii].wres@rrduo[ii]+B[ii].bk)
    B1.fbijiao(y2,yc,0.1)
    return B,yc