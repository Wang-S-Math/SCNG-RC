import numpy as np
from numpy import zeros,zeros_like,arange,array,linspace,ones_like,sum,exp
from numpy.random import rand,randn
class c惠更斯:
    @staticmethod
    def fhuyc3(xx,tt,mu=0.01,p=9):
        (a,q,s,c,r)=(1,1,1,10,1)
        # ng=mu*a*(c*r**2-p)
        xx1=xx[:-2:2]
        xx2=xx[1:-2:2]
        dydt=zeros(8)
        dydt[0]=xx[1]
        dydt[1]=-xx[0]+q*xx[-2]+s*xx[-1]+mu*(a*(c*(r**2-xx[0]**2)-p)*xx[1]-sum(xx1*(1+xx2**2)))
        dydt[2]=xx[3]
        dydt[3]=-xx[2]+q*xx[-2]+s*xx[-1]+mu*(a*(c*(r**2-xx[2]**2)-p)*xx[3]-sum(xx1*(1+xx2**2)))
        dydt[4]=xx[5]
        dydt[5]=-xx[4]+q*xx[-2]+s*xx[-1]+mu*(a*(c*(r**2-xx[4]**2)-p)*xx[5]-sum(xx1*(1+xx2**2)))
        dydt[6]=xx[7]
        dydt[7]=-q*xx[-2]-s*xx[-1]+mu*sum(xx1*(1+xx2**2))
        return dydt

    @staticmethod
    def fhuyc3h(xx,tt,mu=0.01,ng=0.01):
        (a,q,s,c)=(1,1,1,10)
        xx1=xx[:-2:2]
        xx2=xx[1:-2:2]
        dydt=zeros(8)
        dydt[0]=xx[1]
        dydt[1]=-xx[0]-mu*sum(xx1)+ng*xx[1]+q*xx[-2]+s*xx[-1]-mu*c*xx[0]**2*xx[1]-mu*sum(xx1*xx2**2)
        dydt[2]=xx[3]
        dydt[3]=-xx[2]-mu*sum(xx1)+ng*xx[3]+q*xx[-2]+s*xx[-1]-mu*c*xx[2]**2*xx[3]-mu*sum(xx1*xx2**2)
        dydt[4]=xx[5]
        dydt[5]=-xx[4]-mu*sum(xx1)+ng*xx[5]+q*xx[-2]+s*xx[-1]-mu*c*xx[4]**2*xx[5]-mu*sum(xx1*xx2**2)
        dydt[6]=xx[7]
        dydt[7]=-q*xx[-2]-s*xx[-1]+mu*sum(xx1*(1+xx2**2))
        return dydt
    @staticmethod
    def fhuyc3hh(xx,tt,mu=0.01,ng=0.01):
        (a,q,s,c)=(1,1,1,10)
        xx1=xx[:-2:2]
        xx2=xx[1:-2:2]
        dydt=zeros(8)
        dydt[0]=xx[1]
        dydt[1]=-xx[0]-mu*sum(xx1)+ng*xx[1]+q*xx[-2]+s*xx[-1]#-mu*c*xx[0]**2*xx[1]-mu*sum(xx1*xx2**2)
        dydt[2]=xx[3]
        dydt[3]=-xx[2]-mu*sum(xx1)+ng*xx[3]+q*xx[-2]+s*xx[-1]#-mu*c*xx[2]**2*xx[3]-mu*sum(xx1*xx2**2)
        dydt[4]=xx[5]
        dydt[5]=-xx[4]-mu*sum(xx1)+ng*xx[5]+q*xx[-2]+s*xx[-1]#-mu*c*xx[4]**2*xx[5]-mu*sum(xx1*xx2**2)
        dydt[6]=xx[7]
        dydt[7]=xx[-2]+s*xx[-1]+mu*sum(xx1*(1+xx2**2))
        return dydt
    def __init__(self):
        
        import networkx as nx
        import matplotlib.pyplot as plt 
        import timeit
        
        
        from scipy.integrate import odeint,solve_ivp
        from matplotlib.pyplot import plot,legend,figure
        from mpl_toolkits.mplot3d import Axes3D
        
        x0=rand(8)-0.5
        tt=arange(0,500,0.1)
        y1=odeint(c惠更斯.fhuyc3,x0, tt, args=(0.01,9))
        self.y2=odeint(c惠更斯.fhuyc3h,x0, tt, args=(0.01,0.01))
