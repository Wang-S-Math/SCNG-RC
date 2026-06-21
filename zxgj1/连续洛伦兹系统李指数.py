
#%%

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import odeint
from sympy.matrices import Matrix, GramSchmidt
from scipy.integrate import odeint,solve_ivp
def Lorenz(xx,t,params):
    x,y,z=xx
    a,b,c=params
    dydt=[a*(y-x),x*(b-z)-y,x*y-c*z]
    return dydt


def lisanluoze(xx,params,dt):
    x,y,z=xx
    a,b,c=params
    
    dydt=[x+dt*a*(y-x),
          y+dt*(x*(b-z)-y),
          z+dt*(x*y-c*z)]
    return np.array(dydt)
def liluoze_jacobi1(xx,params,dt):
    x,y,z=xx
    a,b,c=params

    dydt=[[1-a*dt,dt*a,0],
          [dt*(b-z),1-dt,-dt*x],
          [dt*y,dt*x,1-c*dt]]
    return np.array(dydt)
def liluoze_jacobi2(xx,params,dt):
    x,y,z=xx
    a,b,c=params

    dydt=[[-a*dt,dt*a,0],
          [dt*(b-z),-dt,-dt*x],
          [dt*y,dt*x,-c*dt]]
    return np.array(dydt)

dt=0.005
t=np.arange(0,100,dt)#(10,28,8/3)(16,45.92,4.0)
x,y,z=1.0,2.0,3.0
yy1=odeint(Lorenz,(x,y,z), t, args=((10,28,8/3),))
def 李指数3():
    #错误
    Q=np.eye(3)
    sle=np.zeros(3)
    n=yy1.shape[0]-1000
    ELE=np.zeros((n,3))
    for i in range(n):
        j=liluoze_jacobi2(yy1[i+1000],(10,28,8/3),dt)
        f1=Q+j@Q
          
        Q, r = qr_positive_r(f1)
        la=np.log(np.abs(np.diagonal(r)))
        sle=sle+la

        ELE[i]=sle/(i*dt)
    print(ELE)
    plt.plot(np.arange(n)*dt,ELE)
    return ELE


ELE=李指数3()
def 李指数3改():
    #错误
    Q=np.eye(3)
    sle=np.zeros(3)
    n=yy1.shape[0]-1000
    ELE=np.zeros((n//100,3))
    for i in range(n):
        j=liluoze_jacobi2(yy1[i+1000],(10,28,8/3),dt)
        f1=Q+j@Q
        if i%100==0:
            Q, r = qr_positive_r(f1)
            la=np.log(np.abs(np.diagonal(r)))
            sle=sle+la
            ELE[i//100]=sle/(i*dt)
        else:
            Q=f1
    print(ELE)
    plt.plot(np.arange(n//100)*100*dt,ELE)
    return ELE


#%%


def Rossler_ly(xx,t,parms=(0.15,0.20,10.0)):
# Rossler吸引子，用来计算Lyapunov指数
    x,y,z=xx[0:3]
    w=xx[3::].reshape(3,3)
    a,b,c=parms
    dydt=np.array([-y-z,x+a*y,b+z*(x-c)])
    jac=np.array([[0,-1,-1],[1,a,0],[z,0,x-c]])
    jw=jac@w
    dydtz=np.concatenate((dydt,jw.reshape(9)))
    return dydtz

def Lorenz_ly(xx,t,parms=(10,28,8/3)):
    x,y,z=xx[0:3]
    w=xx[3::].reshape(3,3)
    a,b,c=parms
    dydt=np.array([a*(y-x),x*(b-z)-y,x*y-c*z])
    jac=np.array([[-a,a,0],[b-z,-1,-x],[y,x,-c]])
    jw=jac@w
    dydtz=np.concatenate((dydt,jw.reshape(9)))
    return dydtz
def qr_positive_r(A):
    # 执行 QR 分解
    Q, R = np.linalg.qr(A)
    
    # 获取 R 对角线的符号（处理可能的零对角线）
    diag_sign = np.sign(np.diag(R))
    diag_sign[diag_sign == 0] = 1  # 避免零对角线影响符号
    
    # 调整 Q 和 R 的符号
    Q_positive = Q * diag_sign  # 每列乘以对应符号
    R_positive = R * diag_sign[:, np.newaxis]  # 每行乘以对应符号
    
    return Q_positive, R_positive
def ThreeGS(V): # V 为3*3向量
    v1 = V[:,0];
    v2 = V[:,1];
    v3 = V[:,2];
    a1 = v1
    a2 = v2-((a1.T@v2)/(a1.T@a1))*a1;
    a3 = v3-((a1.T@v3)/(a1.T@a1))*a1-((a2.T@v3)/(a2.T@a2))*a2;
    return(np.vstack((a1,a2,a3)).T)

xx = np.array([0.1,0.1,0.1]);
Q = np.array([[1 ,0 ,0],[0 ,1, 0],[0 ,0 ,1]])
y=np.concatenate((xx,Q.reshape(9)))
tstart = 0 # % 时间初始值
tstep = 1e-3# % 时间步长
wholetimes = 1e6 # % 总的循环次数
steps = 100  # 每次演化的步数
iteratetimes = int(wholetimes/steps)  # % 演化的次数
mod = np.zeros((3))
lp =  np.zeros((3));
 # 初始化三个Lyapunov指数
Lyapunov1 =  np.zeros((iteratetimes,1));
Lyapunov2 = np. zeros((iteratetimes,1));
Lyapunov3 =  np.zeros((iteratetimes,1));
Lyapunovge=  np.zeros((iteratetimes,3));
Lyapunovge1=  np.zeros((iteratetimes,3));
for i in range(iteratetimes):
    tspan=np.arange(tstart,tstart + tstep*steps, tstep)
    Y = odeint(Lorenz_ly,y,tspan,args=([10,28,10/3.0],));
    # 取积分得到的最后一个时刻的值
    y = Y[99,:]
    y0=y[3:].reshape(3,3)
    
    #  重新定义起始时刻
    tstart = tstart + tstep*(steps-1)
    q,r=qr_positive_r(y0)
    mod1=np.diagonal(r)
    y0 = ThreeGS(y0)
    # 正交化
    #  取三个向量的模
    mod[0] = np.sqrt(y0[:,0].T@y0[:,0])
    mod[1] = np.sqrt(y0[:,1].T@y0[:,1])
    mod[2] = np.sqrt(y0[:,2].T@y0[:,2])
    y0[:,0] = y0[:,0]/mod[0]
    y0[:,1] = y0[:,1]/mod[1]
    y0[:,2] = y0[:,2]/mod[2]
    lp = lp+np.log(np.abs(mod));
    Lyapunovge[i]=mod
    
    Lyapunovge1[i]=mod1
    # print(np.log(np.abs(mod)))
    # 三个Lyapunov指数
    Lyapunov1[i] = lp[0]/(tstart);
    Lyapunov2[i] = lp[1]/(tstart);
    Lyapunov3[i] = lp[2]/(tstart);
    y[3:] = y0.reshape(9)
    # 作Lyapunov指数谱图
plt.figure,
i = np.arange(iteratetimes)
plt.plot(i,Lyapunov1,i,Lyapunov2,i,Lyapunov3)
#%%
dt=0.001
t0=0
tstep=100
ElE=np.zeros((1000,3))
ElE1=np.zeros((1000,3))
sle=np.zeros(3)
y=np.concatenate((xx,Q.reshape(9)))
for ii in range(1000):
    tt=np.arange(t0, t0+ tstep*dt, tstep)
    Y = odeint(Lorenz_ly,y,tspan,args=([10,28,10/3.0],));
    # 取积分得到的最后一个时刻的值
    y = Y[99,:]
    y0=y[3:].reshape(3,3)
    # q,r=qr_positive_r(y0)
    q,r=np.linalg.qr(y0)
    la=np.log(np.abs(np.diagonal(r)))
    ElE1[ii]=la
    sle=sle+la
    #  重新定义起始时刻
    t0 = t0 + dt*(steps-1)
    ElE[ii]=sle/t0
    y0=q
    y[3:] = y0.reshape(9)
    
plt.figure,
i = np.arange(1000)
plt.plot(i,ElE)

















#%%历史
'''


'''