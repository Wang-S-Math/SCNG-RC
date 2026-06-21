import numpy as np




from numpy import zeros,zeros_like,sin,sum
from numpy.random import rand
from scipy.integrate import odeint,solve_ivp


class c仓本模型:
    def __init__(self,t0=0,x0=0,can=0,a=0):
        if isinstance(t0, int):
            self.tt=np.arange(0,100,0.02)
        else:
            self.tt=t0
        self.x0=x0
        self.a=a
        self.yy=zeros_like(self.tt)
        self.can=can
        self.n=-1
        self.k=-1
    def qkrmt(self,x0=0,can=0,a=(-1,-1),qt=1):
        k,n=a
        self
        if n==-1:
            if self.n==-1:
                self.n=10
        else:
            self.n=n
        if k==-1:
            if self.k==-1:
                self.k=0.3
        else:
            self.k=k

        if isinstance(can, int):
            if isinstance(self.can, int): 
                self.can=rand(self.n)-0.5
        else:
            self.can=can 
        if isinstance(x0, int) :
            if isinstance(self.x0,np.ndarray):
                if self.x0.shape[0]!=self.n:
                    self.x0=rand(self.n)
            else:
                self.x0=rand(self.n)
        else:
            self.x0=x0
        if qt==0:
            self.yy=odeint(c仓本模型.fqkuramoto,self.x0, self.tt, args=(self.can,self.k,self.n))
        else:
            self.yy=solve_ivp(c仓本模型.fqtkuramoto,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.k,self.n),rtol=1e-10,atol=1e-11)
        return self.yy 
        

    def krmt(self,x0=0,a=0,can=0,qt=1):
        if isinstance(a, int):
            if isinstance(self.a,int):
                if a==0:
                    a=10
                w=rand(a,a)
                w[w<(3/a)]=1
                w[w<1]=0
                self.a=0.3*w
        else:
            self.a=a
        if isinstance(can, int):
            pass
        else:
            self.can=can 
        if isinstance(self.can, int): 
            self.can=rand(self.a.shape[0])-0.5

        if isinstance(x0, int) :
            if isinstance(self.x0,np.ndarray):
                if self.x0.shape[0]!=self.a.shape[0]:
                    self.x0=rand(self.a.shape[0])
            else:
                self.x0=rand(self.a.shape[0])
        else:
            self.x0=x0

        if qt==0:
            self.yy=odeint(c仓本模型.fkuramoto,self.x0, self.tt, args=(self.can,self.a))
        else:
            self.yy=solve_ivp(c仓本模型.ftkuramoto,(self.tt[0],self.tt[-1]), self.x0, t_eval=self.tt, args=(self.can,self.a),rtol=1e-10,atol=1e-11)
        return self.yy 
    
    @staticmethod     
    def fqkuramoto1(xx,tt,can,K,n):
        dydt=zeros(n)
        for ii in range(n):
            sum1=0
            for jj in range(n):
                sum1=sum1+sin(xx[jj]-xx[ii])
            dydt[ii]=can[ii]+K*sum1
        return dydt
    
    @staticmethod     
    def fqtkuramoto1(tt,xx,can,K,n):
        dydt=zeros(n)
        for ii in range(n):
            sum1=0
            for jj in range(n):
                sum1=sum1+sin(xx[jj]-xx[ii])
            dydt[ii]=can[ii]+K*sum1
        return dydt
    
    @staticmethod     
    def fqtkuramoto(tt,xx,can,K,n):
        dydt=zeros(n)
        for ii in range(n):
            sum1=sum(sin(xx-xx[ii]))
            dydt[ii]=can[ii]+K*sum1
        return dydt
    
    @staticmethod     
    def fqkuramoto(xx,tt,can,K,n):
        dydt=zeros(n)
        for ii in range(n):
            sum1=sum(sin(xx-xx[ii]))
            dydt[ii]=can[ii]+K*sum1
        return dydt 


    @staticmethod   
    def fkuramoto(xx,tt,can,a):
        n=a.shape[0]
        dydt=zeros(n)
        for ii in range(n):
            sum1=0
            for jj in range(n):
                if a[ii][jj]!=0:
                    sum1=sum1+a[ii,jj]*sin(xx[jj]-xx[ii])
            dydt[ii]=can[ii]+sum1
        return dydt
    
   
    @staticmethod   
    def ftkuramoto(tt,xx,can,a):
        n=a.shape[0]
        dydt=zeros(n)
        for ii in range(n):
            sum1=0
            for jj in range(n):
                if a[ii][jj]!=0:
                    sum1=sum1+a[ii,jj]*sin(xx[jj]-xx[ii])
            dydt[ii]=can[ii]+sum1
        return dydt
    
if __name__=='__main__':
    import matplotlib.pyplot as plt 
    from matplotlib.pyplot import plot,legend,figure
    from mpl_toolkits.mplot3d import Axes3D

    A=c仓本模型()
    A.qkrmt(a=(3,100))
    ax=plt.subplot()
    ax.plot(sin(A.yy.y[:4].T))
    print(A.x0)
    print(A.can)
    A.qkrmt(a=(0.1,5))
    plot(sin(A.yy.y[:4].T))
  
    
'''  
from kuramoto import Kuramoto, plot_phase_coherence, plot_activity
graph_nx = nx.erdos_renyi_graph(n=100, p=1) # p=1 -> all-to-all connectivity
adj_mat = nx.to_numpy_array(graph_nx)
model = Kuramoto(coupling=3, dt=0.01, T=10, n_nodes=len(adj_mat))
activity = model.run(adj_mat=adj_mat)
act_mat=model.phase_coherence( activity)
''' 
    
'''x0=np.pi*rand(100)
tt=arange(0,1,0.05)
can=2*rand(100)-1
k=3
n=100
%timeit y1=odeint(c仓本模型.fqkuramoto,x0, tt, args=(can,k,n))
%timeit y2=odeint(c仓本模型.fqkuramoto1,x0, tt, args=(can,k,n))
%timeit y3=solve_ivp(c仓本模型.fqtkuramoto,(tt[0],tt[-1]), x0, t_eval=tt, args=(can,k,n),rtol=1e-8,atol=1e-9)
%timeit y4=solve_ivp(c仓本模型.fqtkuramoto1,(tt[0],tt[-1]), x0, t_eval=tt, args=(can,k,n),rtol=1e-8,atol=1e-9)
4.55 s ± 22.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
276 ms ± 5.09 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
8.61 s ± 30.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
570 ms ± 8.84 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

y1=odeint(c仓本模型.fqkuramoto,x0, tt, args=(can,k,n))
y2=odeint(c仓本模型.fqkuramoto1,x0, tt, args=(can,k,n))
y3=solve_ivp(c仓本模型.fqtkuramoto,(tt[0],tt[-1]), x0, t_eval=tt, args=(can,k,n),rtol=1e-8,atol=1e-9)
y4=solve_ivp(c仓本模型.fqtkuramoto1,(tt[0],tt[-1]), x0, t_eval=tt, args=(can,k,n),rtol=1e-8,atol=1e-9)
plot(sin(y1))
plot(sin(y2))

'''
    
    
    
    
    
    
    