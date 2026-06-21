import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from .llz import clllz1
from matplotlib.pyplot import plot,figure,title
from matplotlib import pyplot as plt


class tezhen:
    def __init__(self,x1=None,x2=None):
        self.t1=None
        self.t2=None
        self.t3=None
        if x1!=None:
            
            if x2!=None:
                self.t1=self.fVPT(x1,x2)
    def fVPT(self,x1,x2,bi=0.1):
        if x1.ndim!=x2.ndim:
            print('维度不同')
            return 0
        else:
            jieguo=0
            if x1.ndim==1:
                dda=bi*np.var(x2)
                if x1.shape[0]<x2.shape[0]:
                    nn=x1.shape[0]
                else:
                    nn=x2.shape[0]
                ce=x1[:nn]-x2[:nn]
                absce=np.abs(ce)
                for ii in range(nn):
                    if absce[ii]>dda:
                        print(ii)
                        break
                    jieguo=ii+1
            if x1.ndim==2:
                dda=bi*np.var(x2,axis=0)
                if x1.shape[0]<x1.shape[1]:
                    x1=x1.T  
                if x2.shape[0]<x2.shape[1]:
                    x2=x2.T
                if x1.shape[0]<x2.shape[0]:
                    nn=x1.shape[0]
                else:
                    nn=x2.shape[0]
                ce=x1[:nn]-x2[:nn]
                absce=np.abs(ce)
                for ii in range(nn):
                    if (absce[ii]>dda).any():
                        print(ii)
                        break
                    jieguo=ii+1
        return(jieguo)
    def flizhishu1(self,x1):
        pass

class lzhishu:
    def __init__(self,xx=0,cc=150,mts=500,ed=5,td=2):
        self.xx=xx
        if isinstance(xx,np.ndarray):
            if xx.ndim==1:
                self.qlzsh1(xx,cc,mts,ed,td)
            else:
                self.qlzsh2(xx,cc,mts)
        
    def qlzsh1(self,xx,cc=150,mts=500,ed=5,td=2):
        self.phsp=self.f1(xx,ed,td)
        self.tstp,self.ld,self.ld1,self.ldcw=self.jsdel(self.phsp, mts)
        self.lzhsh=self.nihe22(self.tstp[:cc],self.ld[:cc])
        self.lzhsh1=self.nihe(self.tstp[:cc],self.ld1[:cc])
        print(self.lzhsh[0])
        return(self.lzhsh)
    
    def qlzsh2(self,xx,cc=150,mts=500):
        self.tstp,self.ld,self.ld1,self.ldcw=self.jsdel(xx,mts)
        self.lzhsh=self.nihe22(self.tstp[:cc],self.ld[:cc])
        self.lzhsh1=self.nihe(self.tstp[:cc],self.ld1[:cc])
        print(self.lzhsh[0])
        return(self.lzhsh)
    def qlzsh3(self,xx,cc=150,mts=500, metric='euclidean'):
        ddt = squareform(pdist(xx,metric=metric))
        log_divergences2=[]
        TT=np.arange(1,mts)
        for tt in TT:
            sumdg=0
            for ii in range(0,ddt.shape[0]-tt):
                sumdg+=np.log(ddt[ii,ii+tt])
            log_divergences2.append(sumdg/(ddt.shape[0]-tt))
        return(TT,log_divergences2)
    def qitlz1(self,xx):
        import nolds
        # nolds.lyap_e(xx)
        # nolds.lyap_r(xx)
        return (nolds.lyap_e(xx),nolds.lyap_r(xx))
    def qitlz2(self,xx,dt=0.01,window = 60,tau = 13,dim = [5]):
        from nolitsa import lyapunov
        sample = dt
        #window: Choose appropriate Theiler window.
        # Time delay:tau
        # Embedding dimension:dim
        
        d = lyapunov.mle_embed(xx, dim=dim, tau=tau, maxt=300, window=window)[0]
        t = np.arange(300)
        self.tstp2=t
        self.ld2=d
        self.lzhsh2=self.nihe22(t, d)
        
        plt.title('Maximum Lyapunov exponent for the Lorenz system')
        plt.xlabel(r'Time $t$')
        plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
        plt.plot(sample * t, d)
        plt.plot(sample * t, sample * t * self.lzhsh2, '--')

        plt.show()
        return self.lzhsh2

    
    def gainihe(self,cc,c0=0,leix=0):
        if c0!=0:
            fd=c0
            c0=cc
            cc=fd
        if leix==0:
            self.lzhsh=self.nihe22(self.tstp[c0:cc],self.ld[c0:cc])
            print(self.lzhsh[0])
        elif leix==1:
            self.lzhsh1=self.nihe(self.tstp[c0:cc],self.ld1[c0:cc])
            print(self.lzhsh1)
        elif leix==2:
            self.lzhsh2=self.nihe22(self.tstp2[c0:cc],self.ld2[c0:cc])
            print(self.lzhsh2)
        else:
            pass
            return self.lzhsh2
    def duibi(self,leix=0):
        if leix==0:
            plot(self.tstp,self.ld)
            plot(self.tstp,self.lzhsh[0]*self.tstp+self.lzhsh[1])
        elif leix==1:
            plot(self.tstp,self.ld1)
            plot(self.tstp,self.lzhsh1*self.tstp)
        elif leix==2: 
            plot(self.tstp2,self.ld2)
            plot(self.tstp2,self.lzhsh2[0]*self.tstp2+self.lzhsh2[1])
        else:
            pass
    def f1(self,xx,ed,td):
        ##reconstruct_phase_space
        ##维度嵌入，时间延迟为td,嵌入维度为ed
        N=len(xx)
        phsp=np.zeros((N-(ed-1)*td,ed))
        for i in range(ed):
            phsp[:,i]=xx[i*td:N-(ed-1-i)*td]
        return phsp
    
    def f2(self,phsp, metric='euclidean'):
        #find_nearest_neighbors
        dt = squareform(pdist(phsp, metric=metric))#距离
        np.fill_diagonal(dt, np.inf)  # 避免自匹配
        
        nid = np.argmin(dt, axis=1) #最近距离序号
        ndt = np.min(dt, axis=1) #最近距离
        return nid, ndt
    def f2g1(self,phsp, metric='euclidean'):
        #find_nearest_neighbors
        dt = squareform(pdist(phsp, metric=metric))#距离
        np.fill_diagonal(dt, np.inf)  # 避免自匹配
        dt1=dt
        ll=dt.shape[0];
        for ii in range(100):
            dt1[ii,:ii+100]=100
        for ii in range(100,ll-100):
            dt1[ii,ii-100:ii+100]=100
        for ii in range(ll-100,ll):
            dt1[ii,ii-100:]=100
        nid = np.argmin(dt1, axis=1) #最近距离序号
        ndt = np.min(dt1, axis=1) #最近距离
        return nid, ndt
    
    def f2g0(self,phsp, metric='euclidean'):
        #find_nearest_neighbors
        dt = squareform(pdist(phsp, metric=metric))#距离
        ll=dt.shape[0];nid=np.zeros(ll);ndt=np.zeros(ll)
        for ii in range(100):
            id0=ii
            dt0=dt[ii,ii+1]
            for jj in range(ii+100,ll):
                if dt[ii,jj]<dt0:
                    id0=jj
                    dt0=dt[ii,jj]
            nid[ii]=id0
            ndt[ii]=dt0
        for ii in range(100,ll-100):
            id0=ii
            dt0=dt[ii,ii+1]
            for jj in range(0,ii-99):
                if dt[ii,jj]<dt0:
                    id0=jj
                    dt0=dt[ii,jj]
            for jj in range(ii+100,ll):
                if dt[ii,jj]<dt0:
                    id0=jj
                    dt0=dt[ii,jj]
            nid[ii]=id0
            ndt[ii]=dt0
        for ii in range(ll-100,ll):
            id0=ii
            dt0=dt[ii,ii-1]
            for jj in range(0,ii-100):
                if dt[ii,jj]<dt0:
                    id0=jj
                    dt0=dt[ii,jj]
            nid[ii]=id0
            ndt[ii]=dt0       
        return nid, ndt
    
    def jsdel(self,phsp,max_time_steps):
        ## 求随时间距离
        
        nid, ndt = self.f2g1(phsp)
        tstp = np.arange(1, max_time_steps + 1)
        log_divergencescw = []
        log_divergences1 = []
        log_divergences2 = []
        ff=len(phsp)
        for t in tstp:
            c0t=phsp[t:] #原来值t时刻后值
            c1t=(nid+t)[:-t]  #邻接值t时刻后序号
            c2t=c1t[c1t<ff]   #邻接值t时刻后序号小于最大值的序号
            y1g=phsp[c2t]    #邻接值t时刻后序号小于最大值的值
            y0g=c0t[c1t<ff]   #原来值对应的t时刻后值
            divergences = np.linalg.norm(y1g - y0g, axis=1) 
            divergences1=divergences.copy()  
            divergences2=divergences.copy()
            divergences /= ndt[c2t-t]   #除以初始值不对
            divergences1 /= ndt[:-t][c1t<ff]  #除以修改初始值

            log_divergencescw.append(np.mean(np.log(divergences[divergences> 0])))#除以初始值不对
            log_divergences1.append(np.mean(np.log(divergences1[divergences1 > 0]))) #正确
            log_divergences2.append(np.mean(np.log(divergences2[divergences2 > 0])))  #不除初始值
        return tstp,np.array(log_divergences1),np.array(log_divergences2),np.array(log_divergencescw)

    ## 最优化拟合
    def nihe(self,tstp,log_divergences):
        def linear_model(tt, mLLE):
            return mLLE * tt
        params, _ = curve_fit(linear_model, tstp, log_divergences)
        return params
        
    def nihe1(self,tstp,log_divergences):
        def linear_model(tt, mLLE,l2):
            return mLLE * tt+l2
        params, _ = curve_fit(linear_model, tstp, log_divergences) 
        return params
    def nihe22(self,tstp,log_divergences):
        #log_divergences=a1*tstp+a2
        mean1=np.mean(tstp);mean2=np.mean(log_divergences)
        c1=tstp-mean1
        l1=np.sum(c1*(log_divergences-mean2))
        l2=np.sum(c1**2)
        a1=l1/l2
        b1=mean2-a1*mean1
        return a1,b1
    def zynh(self,tstp,yy):
        ll=len(tstp)
        l1=[];l2=[];l3=[];
        c0=1
        if ll<101:
            return (self.nihe22(tstp,yy))
        else:
            for ii in range(0,ll-101):
                for jj in range(100+ii,ll):
                    a,b=self.nihe22(tstp[ii:jj],yy[ii:jj])
                    y1=a*tstp[ii:jj]+b
                    ce=np.mean((yy[ii:jj]-y1)**2)
                    if ce<c0:
                        c0=ce
                        ww=((ii,jj),(a,b))
                    l1.append([ii,jj])
                    l2.append((a,b))
                    l3.append(ce)
            return(ww,(l1,l2,l3))     
        
class guanlianwei:
    def __init__(self,x1=None):
        # self.r=np.arange(-2,6,0.1)
        # self.rr=np.exp(self.r)
        self.rr=np.arange(0.01,4,0.05)
        self.r=np.log(self.rr)
        self.cc=np.zeros(len(self.rr))
        self.guanlw=None

    def g1(self,x1):
        self.jisuan(x1)
        self.jisuan2()
        self.duibi()

    def g2(self,x1):
        self.jisuang(x1)
        self.jisuan2()
        self.duibi()
    
    def g3(self,x3):
        import nolds
        return(nolds.corr_dim(yy,7))
    def huatu1(self):
        plot(self.rr, self.cc)
    def huatu2(self):
        plot(np.log(self.rr[self.cc>0]), np.log(self.cc[self.cc>0]))
    def huatu1t(self):
        plot(self.rr, np.exp(self.guanlw[1])*self.rr**self.guanlw[0])
    def huatu2t(self):
        plot(self.r, self.guanlw[0]*self.r+self.guanlw[1])
    def jisuan2(self,c2=0,c1=0):
        if c2==0:
            c2=len(self.rr)
        if c1!=0:
            fd=c1
            c1=c2
            c2=fd
        ccg=self.cc[self.cc>0]
        rrg=self.rr[self.cc>0]
        l1=np.log(rrg)
        l2=np.log(ccg)
        self.guanlw=np.polyfit(l1[c1:c2],l2[c1:c2],1)
        return self.guanlw        
        
    def duibi(self):
        figure(1)
        self.huatu1()
        self.huatu1t()
        figure(2)        
        self.huatu2()
        self.huatu2t()
        
    def jisuan(self,xx,ed=5,td=1,tzh=False):
        self.yy=self.f3(xx,ed,td,tzh)
        for ii in range(len(self.rr)):
            self.cc[ii]=self.glwei1(self.yy,self.rr[ii])
        return self.cc
    def jisuang(self,xx,ed=5,td=1,tzh=False):
        self.yy=self.f3(xx,ed,td,tzh)
        ddtt=self.glwei1g1(self.yy,metric='euclidean')
        self.cc=self.glwei1g2(ddtt,self.rr)
        return self.cc
    
    def glwei1(self,yy,e,metric='euclidean'):
        l1,l2=yy.shape
        ll2=l1**2
        sum1 = -1
        for ii in range(l1):
            xx0=yy[ii,:]
            xxc=np.abs(xx0-yy)
            xxd=np.max(xxc,axis=1)
            ss=sum(e>=xxd)
            sum1=sum1+ss
        cc=sum1/ll2
        return(cc)
    def glwei1g1(self,yy,metric='euclidean'):
        dtt = squareform(pdist(yy, metric=metric))#距离
        return(dtt)
    def glwei1g2(self,dtt,ee):
        l1,l2=dtt.shape
        ll2=(l1-1)**2
        cc=np.zeros(len(ee))
        for ii,jj in enumerate(ee):
            sum1=np.sum(dtt<=jj)-l1
            cc[ii]=sum1/ll2
        return(cc)

    def f1(self,xx,ed=5,td=1):
            ##reconstruct_phase_space
            ##维度嵌入，时间延迟为td,嵌入维度为ed
        N=len(xx)
        yy=np.zeros((N-(ed-1)*td,ed))
        for i in range(ed):
            yy[:,i]=xx[i*td:N-(ed-1-i)*td]
        return yy
    def f2(self,xx,ed=5,td=1):
            ##reconstruct_phase_space
            ##维度嵌入，时间延迟为td,嵌入维度为ed
        l1,l2=xx.shape
        yy=np.zeros((l1-(ed-1)*td,l2*ed))
        ij=0
        for i in range(ed):
            for j in range(l2):
                yy[:,ij]=xx[i*td:l1-(ed-1-i)*td,j]
                ij+=1
        return yy
    def f3(self,xx,ed=5,td=1,tzh=False):
        if xx.ndim==1:
            yy=self.f1(xx,ed,td)
        if xx.ndim==2:
            if tzh==False:
                yy=xx
            else:
                yy=self.f2(xx,ed,td)
        return(yy)

class c极值:
    def __init__(self,xx,qt=None):
        if xx.ndim==1:
            self.c=self.fjxzh(xx)
        if xx.ndim==2:
            self.c=[]
            if qt==None:
                if xx.shape[0]<xx.shape[1]:
                    xx=xx.T
            elif qt==0:
                xx=xx.T
            else:
                pass
            for ii in range(xx.shape[1]):
                self.c.append(self.fjxzh(xx[:,ii]))
    def fjxzh(self,xx):
        c=[]
        for ii in range(1,xx.shape[0]-1):
            if xx[ii-1]<xx[ii]<xx[ii+1]:
                c.append(xx[ii])
        return c
    


if __name__=='__main__':
    A=clllz1()
    A.tt=np.arange(200,step=0.01)
    A.can=(16,45.92,4.0)
    A.dange1()
    yy=A.yy[1000:6000,:]
    B=guanlianwei()
    B.g2(yy)
    B.guanlw[0]

    
    
    # B=lzhishu(yy,150)
    # aa,bb=B.zynh(B.tstp,B.ld)
    # B.gainihe(30,200)
    # B.duibi()
    
    # B.qlzsh1(yy[:,0])
    # B.duibi
    # B1=lzhishu(yy[:,0],150)
    # B1.duibi()
    # B1.gainihe(30,200)
    # B1.duibi()
    
    # B1.qitlz1(yy[:,0])
    
    # B1.qitlz2(yy[:,0])
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    