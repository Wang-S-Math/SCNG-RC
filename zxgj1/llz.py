# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:48:06 2024

@author: lenovo
"""
import numpy as np

from numpy import zeros,zeros_like,sin
from numpy.random import rand
from scipy.integrate import odeint,solve_ivp

class clllz1:

    def __init__(self,t=0,x0=0,can=(10,28,3),a=0):

        if isinstance(t, int):
            self.tt=np.arange(0,50,0.01 )
        else:
            self.tt=t
        self.x0=x0
        self.a=a
        self.yy=zeros_like(self.tt)
        self.can=can
    def mllz(self,x0=0,qt=0):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(3)
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(clllz1.f1luolunz, self.x0, self.tt, args=self.can)
        elif qt==3:
            self.yy=odeint(clllz1.f2luolunz, self.x0, self.tt, args=self.can)
        return self.yy  

        
    def mc3llz(self,x0=0,qt=0):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(9)
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(clllz1.f2ouheluolunz3, self.x0, self.tt,args=self.can)
        elif qt==1:
            self.yy=solve_ivp(clllz1.f2touheluolunz3, (self.tt[0],self.tt[-1]),self.x0, t_eval=self.tt,args=self.can,rtol=1e-10,atol=1e-11)
        return self.yy 
    
    
    def mcnllzy_x_x(self,x0=0,qt=0):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(3*self.a.shape[0])
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(clllz1.f2oullzny_x_x, self.x0, self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        elif qt==3:
            self.yy=odeint(clllz1.f2oullzny_x_x1, self.x0, self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        else:
            self.yy=solve_ivp(clllz1.f2toullzny_x_x,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        return self.yy  
    def mcnllzy_x_xsin(self,x0=0,qt=0):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(3*self.a.shape[0])
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(clllz1.f2oullzny_x_xsin, self.x0, self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        elif qt==3:
            self.yy=odeint(clllz1.f2oullzny_x_xsin1, self.x0, self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        else:
            pass
            # self.yy=solve_ivp(clllz1.f2toullzny_x_x,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        return self.yy  
    def mcnllzx_x_x(self,x0=0,qt=1):
        if isinstance(x0, int) :
            if isinstance(self.x0, int):
                self.x0=rand(3*self.a.shape[0])
        else:
            self.x0=x0
        if qt==0:
            pass
        else:
            self.yy=solve_ivp(clllz1.f2toullznx_x_x,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11) 
        return self.yy

    @staticmethod
    def f1luolunz(xx,tt,a,b,c):
        x,y,z=xx
        dydt=[a * (y - x),-y+b*x-x*z,x * y-c*z]
        return dydt
    @staticmethod
    def f2luolunz(xx,tt,a,b,c):
        dydt=zeros(3)
        dydt[0]=a * (xx[1] - xx[0])
        dydt[1]=-xx[1]+b*xx[0]-xx[0]*xx[2]
        dydt[2]=xx[0] * xx[1]-c*xx[2]
        return dydt
    @staticmethod
    def f2ouheluolunz3(xx,tt,a,b,c):
        dydt=zeros(9)
        dydt[0]=a * (xx[1] - xx[0])
        dydt[1]=-xx[1]+b*xx[6]-xx[0]*xx[2]
        dydt[2]=xx[0] * xx[1]-c*xx[2]
        dydt[3]=a * (xx[4] - xx[3])
        dydt[4]=-xx[4]+b*xx[0]-xx[3]*xx[5]
        dydt[5]=xx[3] * xx[4]-c*xx[5]
        dydt[6]=a * (xx[7] - xx[6])
        dydt[7]=-xx[7]+b*xx[3]-xx[6]*xx[8]
        dydt[8]=xx[6] * xx[7]-c*xx[8]
        return dydt
    
    @staticmethod
    def f2touheluolunz3(tt,xx,a,b,c):
        dydt=zeros(9)
        dydt[0]=a * (xx[1] - xx[0])
        dydt[1]=-xx[1]+b*xx[6]-xx[0]*xx[2]
        dydt[2]=xx[0] * xx[1]-c*xx[2]
        dydt[3]=a * (xx[4] - xx[3])
        dydt[4]=-xx[4]+b*xx[0]-xx[3]*xx[5]
        dydt[5]=xx[3] * xx[4]-c*xx[5]
        dydt[6]=a * (xx[7] - xx[6])
        dydt[7]=-xx[7]+b*xx[3]-xx[6]*xx[8]
        dydt[8]=xx[6] * xx[7]-c*xx[8]
        return dydt
    
    @staticmethod
    def f2toullzny_x_x(tt,xx,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        for ii in range(n):
            sum1=0
            for jj in range(ii):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            for jj in range(ii+1,n):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]+sum1
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt
    @staticmethod
    def f2oullzny_x_x(xx,tt,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        for ii in range(n):
            sum1=0
            for jj in range(ii):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            for jj in range(ii+1,n):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
                
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]+sum1
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt
    @staticmethod
    def f2oullzny_x_xsin(xx,tt,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        for ii in range(n):
            sum1=0
            for jj in range(ii):
                sum1=sum1+w[ii,jj]*sin(xx[3*jj]-xx[3*ii])
            for jj in range(ii+1,n):
                sum1=sum1+w[ii,jj]*sin(xx[3*jj]-xx[3*ii])
                
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]+sum1
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt
    @staticmethod
    def f2oullzny_x_xsin1(xx,tt,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        xd=xx[::3]
        for ii in range(n):
            xc=np.dot(w[ii],sin(xd-xd[ii])) 
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]+xc
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt    
    @staticmethod
    def f2oullzny_x_x1(xx,tt,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        xd=xx[::3]
        xc=zeros(n)
        for ii in range(n):
            xc[ii]=np.dot(w[ii],(xd-xd[ii])) 
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]+xc[ii]
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt

    @staticmethod
    def f2toullznx_x_x(tt,xx,sets,w):
        a,b,c=sets
        n=w.shape[0]
        dydt=zeros(3*n)
        for ii in range(n):
            sum1=0
            for jj in range(ii):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
            for jj in range(ii+1,n):
                sum1=sum1+w[ii,jj]*(xx[3*jj]-xx[3*ii])
                
            dydt[3*ii]=a*(xx[3*ii+1] - xx[3*ii])+sum1
            dydt[3*ii+1]=-xx[3*ii+1]+b*xx[3*ii]-xx[3*ii]*xx[3*ii+2]
            dydt[3*ii+2]=xx[3*ii] * xx[3*ii+1]-c*xx[3*ii+2]
        return dydt
   

if __name__=='__main__':
    import matplotlib.pyplot as plt 
    from mpl_toolkits.mplot3d import Axes3D
    A=clllz1()
    # a=np.array([[0., 0., 1., 1., 0.],
    #    [1., 0., 0., 0., 0.],
    #    [1., 1., 0., 1., 1.],
    #    [1., 0., 0., 1., 0.],
    #    [1., 0., 1., 0., 0.]])
    # A.a=a
    # A.ouhe2()
    A.sanget2()
    # np.save('yy.npy',A.yy)
    # print(A.sanget2())
    plt.plot(A.yy.y[:,:3])
    plt.show
    # def B():
        #     A.dange2()
        # t = timeit.repeat('B()','from __main__ import B', number=100, repeat=5)
        # print(t)

    




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        