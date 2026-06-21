# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:48:06 2024

@author: lenovo
"""
import numpy as np

from numpy import zeros,zeros_like,array
from numpy.random import rand,randn
from scipy.integrate import odeint,solve_ivp


class cllsl:
    def __init__(self,t0=0,x0=0,can=(0.1,0.2,10),a=0):
        if isinstance(t0, int):
            self.tt=np.arange(0,100,0.02)
        else:
            self.tt=t0
        self.x0=x0
        self.a=a
        self.yy=zeros_like(self.tt)
        self.can=can

    def rossler(self,x0=0,qt=1):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(3)
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(cllsl.frossler, self.x0, self.tt, args=self.can)
        else:
            self.yy=solve_ivp(cllsl.ftrossler,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=self.can,rtol=1e-10,atol=1e-11)
        return self.yy
    
    def crslx_x_x(self,x0=0,a=0,can=0,qt=0):
        if isinstance(a, int):
            if isinstance(self.a,int):
                if a==0:
                    a=10
                w=rand(a,a)
                w[w<(4/a)]=1
                w[w<1]=0
                self.a=w      
        else:
            self.a=a
        if isinstance(can, int):
            pass
        else:
            self.can=can       
        if isinstance(x0, int) :
            if isinstance(self.x0, np.ndarray):
                if self.x0.shape[0]!=3*self.a.shape[0]:
                    self.x0=rand(3*self.a.shape[0])
            else:
                self.x0=rand(3*self.a.shape[0])
        else:
            self.x0=x0  
            
        if qt==0:
            self.yy=odeint(cllsl.fcrosx_x_x,self.x0, self.tt, args=(self.can,self.a))
        else:
            self.yy=solve_ivp(cllsl.ftcrosx_x_x,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11)
        return self.yy  
     
    @staticmethod
    def fcrosx_x_x(xx,tt,can,w):
        a,b,c=can
        n=w.shape[0]
        dydt=zeros(3*n)
        xd=xx[::3]
        xc=zeros(n)
        for ii in range(n):
            xc[ii]=np.dot(w[ii],(xd-xd[ii]))
            dydt[3*ii]=-xx[3*ii+1]-xx[3*ii+2]+xc[ii]
            dydt[3*ii+1]=xx[3*ii+0]+a*xx[3*ii+1]
            dydt[3*ii+2]=b+xx[3*ii+2]*(xx[3*ii+0]-c)
        return dydt
    
    @staticmethod
    def ftcrosx_x_x(tt,xx,can,w):
        a,b,c=can
        n=w.shape[0]
        dydt=zeros(3*n)
        xd=xx[::3]
        xc=zeros(n)
        for ii in range(n):
            xc[ii]=np.dot(w[ii],(xd-xd[ii]))
            dydt[3*ii]=-xx[3*ii+1]-xx[3*ii+2]+xc[ii]
            dydt[3*ii+1]=xx[3*ii+0]+a*xx[3*ii+1]
            dydt[3*ii+2]=b+xx[3*ii+2]*(xx[3*ii+0]-c)
        return dydt
    
    @staticmethod
    def fcrosx_x_x1(xx,tt,can,w):
        a,b,c=can
        n=w.shape[0]
        dydt=zeros(3*n)
        for ii in range(n):
            sum1=0
            for jj in range(ii):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            for jj in range(ii+1,n):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            dydt[3*ii]=-xx[3*ii+1]-xx[3*ii+2]+sum1
            dydt[3*ii+1]=xx[3*ii+0]+a*xx[3*ii+1]
            dydt[3*ii+2]=b+xx[3*ii+2]*(xx[3*ii+0]-c)
        return dydt           
        

    @staticmethod
    def frossler(xx,tt,a,b,c):
        dydt=zeros(3)
        dydt[0]=-xx[1] - xx[2]
        dydt[1]=xx[0]+a*xx[1]
        dydt[2]=b+xx[2] * (xx[0]-c )
        return dydt
    
    @staticmethod
    def ftrossler(tt,xx,a,b,c):
        dydt=zeros(3)
        dydt[0]=-xx[1] - xx[2]
        dydt[1]=xx[0]+a*xx[1]
        dydt[2]=b+xx[2] * (xx[0]-c )
        return dydt
def fjxzh(xx):
    c=[]
    for ii in range(1,xx.shape[0]-1):
        if xx[ii-1]<xx[ii]>xx[ii+1]:
            c.append(xx[ii])
    return array(c)
def jxzh(xx,qt=None):
    if xx.ndim==1:
        c=fjxzh(xx)
    if xx.ndim==2:
        c=[]
        if qt==None:
            if xx.shape[0]<xx.shape[1]:
                xx=xx.T
        elif qt==0:
                xx=xx.T
        else:
            pass
        for ii in range(xx.shape[1]):
            c.append(fjxzh(xx[:,ii]))
    return c
if __name__=='__main__':
    import matplotlib.pyplot as plt 
    from matplotlib.pyplot import plot,legend,figure
    from mpl_toolkits.mplot3d import Axes3D
    A=cllsl()
    
    A.rossler()
    
    A.can=(0.1,0.2,10)
    A.rossler()
    figure()
    ax = plt.axes(projection='3d')
    # ax.plot(A.yy[:,0], A.yy[:,1],A.yy[:,2])
    ax.plot(A.yy.y[0,:], A.yy.y[1,:],A.yy.y[2,:])
    ##
    
    A.crslx_x_x()
    
    '''
    cc=jxzh(A.yy)
    plot(0.1*ones_like(cc[0]),cc[0],'.')
    # plt.show()
    figure()
    hc=[]
    for ii in np.arange(0.02,1,0.02):
        A.can=(ii,0.2,10)
        A.rossler()
        cc=jxzh(A.yy)
        hc.append(cc)
        plot(ii*ones_like(cc[0]),cc[0],'.')
        
    for ii in np.arange(len(hc)):
        plot(0.02*ii*ones_like(hc[ii][2]),hc[ii][2],'.')
    '''
    ##测试速度
'''
    w=rand(10,10)
w[w<0.4]=1
w[w<1]=0
x0=rand(30)
tt=np.arange(0,100,0.02)
%timeit y1=odeint(cllsl.fcrosx_x_x,x0, tt, args=((0.1,0.3,10),w))
%timeit y2=odeint(cllsl.fcrosx_x_x1,x0, tt, args=((0.1,0.3,10),w))
%timeit y3=odeint(cllsl.fcrosx_x_x0,x0, tt, args=((0.1,0.3,10),w))
200 ms ± 2.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
247 ms ± 15.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
194 ms ± 1.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit y1=odeint(cllsl.fcrosx_x_x,x0, tt, args=((0.1,0.3,10),w))
%timeit y2=odeint(cllsl.fcrosx_x_x1,x0, tt, args=((0.1,0.3,10),w))
%timeit y3=odeint(cllsl.fcrosx_x_x0,x0, tt, args=((0.1,0.3,10),w))
189 ms ± 816 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
243 ms ± 8.11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
183 ms ± 1.77 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
y1=odeint(cllsl.fcrosx_x_x,x0, tt, args=((0.1,0.3,10),w))
y2=odeint(cllsl.fcrosx_x_x1,x0, tt, args=((0.1,0.3,10),w))
y3=odeint(cllsl.fcrosx_x_x0,x0, tt, args=((0.1,0.3,10),w))
plot(y1[:,1])
plot(y2[:,1])
plot(y3[:,1])
'''