import numpy as np
from numpy import var,mean,std,sqrt
from numpy.linalg import norm
class Vpt:
    @staticmethod
    def vpt(x1,x2):
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]        
        dda=np.array([5.8,8,6.9])
        ce=x1[:nn]-x2[:nn]
        absce=np.abs(ce)
        for ii in range(nn):
            jieguo=ii
            if (absce[ii]>dda).any():
                print(ii)
                break
        return(jieguo)
    @staticmethod
    def vpt1(x1,x2,bi=0.15):
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]        
        dda=bi*var(x2,axis=0)
        ce=x1[:nn]-x2[:nn]
        absce=np.abs(ce)
        for ii in range(nn):
            jieguo=ii
            if (absce[ii]>dda).any():
                print(ii)
                break
        return(jieguo)
    
    @staticmethod
    def vpt2(x1,x2,bi=0.15):
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]
        dda=bi*(np.max(x2,0)-np.min(x2,0))
        ce=x1[:nn]-x2[:nn]
        absce=np.abs(ce)
        for ii in range(nn):
            jieguo=ii
            if (absce[ii]>dda).any():
                print(ii)
                break
            
        return(jieguo)
    @staticmethod
    def vpt3(x1,x2,bi=0.5):
        '''论文高阶格兰杰,全局预测'''
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]
        stdx=std(x2,0)

        ce=((x1[:nn]-x2[:nn])/stdx)**2
        rmse=sqrt(mean(ce,axis=1))
        for ii in range(nn):
            jieguo=ii
            if rmse[ii]>bi:
                print(ii)
                break
            
        return(jieguo)
    @staticmethod
    def vpt4(x1,x2,bi=0.4):
        '''基于数据的'''
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]
        ce=norm(x1[:nn]-x2[:nn],axis=1)
        dda=bi*mean(norm(x2-mean(x2,0),axis=1))
        for ii in range(nn):
            jieguo=ii
            if (ce[ii]>dda):
                print(ii)
                break
            
        return(jieguo)
    @staticmethod
    def vpt5(x1,x2,bi=0.4):
        if x1.shape[0]<x1.shape[1]:
            x1=x1.T  
        if x2.shape[0]<x2.shape[1]:
            x2=x2.T
        if x1.shape[0]<x2.shape[0]:
            nn=x1.shape[0]
        else:
            nn=x2.shape[0]
        ce=norm(x1[:nn]-x2[:nn],axis=1)
        dda=bi*mean(norm(x2,axis=1))
        for ii in range(nn):
            jieguo=ii
            if (ce[ii]>dda):
                print(ii)
                break
            
        return(jieguo)
    def vpt6(x1,x2,bi=0.15):
        pass
if __name__=='__main__':
    t=np.arange(0,100,0.01)    
    v2=np.sin(t)
    v21=np.sin(t+0.5)
    v22=np.sin(t+1)
    v1=v2+0.1*t*(np.random.rand(10000)-0.5)
    v11=v21+0.05*t*(np.random.rand(10000)-0.5)
    v12=v22+0.08*t*(np.random.rand(10000)-0.5)
    x1=np.vstack((v1,v11,v12)).T
    x2=np.vstack((v2,v21,v22)).T
    A=Vpt()
    c1=A.vpt1(x1, x2)
    c2=A.vpt2(x1, x2)
    c3=A.vpt3(x1, x2)
    c4=A.vpt4(x1, x2)
    c5=A.vpt5(x1, x2)
    c6=A.vpt6(x1, x2)













