import numpy as np
#import scipy.special  
import seaborn as sns
import matplotlib.pyplot as plt
#from yapf.yapflib.yapf_api import FormatCode
from numpy import linalg as LA

class functions:
#__init__(self,sig,c)
#kernel(self,r_i,r_r)
#computeV(mapping,currentposition)
#overlap(matA,matB)
#distance(r_1,r_2)
#transfer(h,th,g)
#positive_mean(v,th)
#fix_parameters(V1,h1,th,a,a2)
#dynamic(position,J,mapping)
#random_trajectory(self,L,n,t)
#ordered_trajectory(self,L,n,t)
#sample_regular_trajectory(self,L,n,epochs)

    
    def __init__(self,sig,c):
        self.sig=sig
        self.c=c

    def kernel(self,r_i,r_r):#r_i=position on the grid,r_r=rat position,sig=standard deviation,c=cutoff
        V=0.0  
        dx=0
        dy=0
        d=0
        dx=np.abs(r_i[0]-r_r[0])
        dy=np.abs(r_i[1]-r_r[1])
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        if d<=self.c:
            V=np.exp(-0.5*pow(d/self.sig,2))
        else:
            V=0    
        return V

    def computeV(self,mapping,currentposition): #computes the activity of the network depending on the current position of the external input
        N=len(mapping)    
        V=np.zeros(N)
        for i in range(N):
            V[i]=self.kernel(mapping[i],currentposition)
        return V

    def overlap(self,matA,matB):
        m=0
        for i in range(matA.shape[0]):
            for j in range(matA.shape[0]):
                m=m+matA[i][j]*matB[i][j]
        m=m/float(LA.norm(matA)*LA.norm(matB))
        return m

    def distance(self,r_1,r_2):
        dx=0
        dy=0
        d=0
        dx=np.abs(r_1[0]-r_2[0])
        dy=np.abs(r_1[1]-r_2[1])
        d=np.sqrt(pow(dx,2)+pow(dy,2))
        return d

#transfer function
    def transfer(self,h,th,g):
        if (h-th)>0:
            return g*(h-th)
        else:
            return 0.0
    
    def positive_mean(v,th):
        return np.mean([x-th for x in v if x-th>0])

#interaction kernel

    def fix_parameters(self,V1,h1,th,a,a2):
        #for threshold fixing
        b=0.01
        sigma_mean=0.001
        maxiter=10000000
        lmbd=10
        fixed=False
        it=0
        while (not fixed) and it<maxiter:
            th=th+b*(pow(np.mean(V1),2)/np.mean(pow(V1,2))-a2)
            V1=np.asarray(list(map(lambda h1: self.transfer(h1,th,self.dynamic.g),h1)))
            fixed=(np.abs((pow(np.mean(V1),2)/np.mean(pow(V1,2))-a2))/a2 <= sigma_mean)
    #g=a/positive_mean(h1,th)
        if it>=maxiter:
            print("Iter bound reached")
        return th

    def dynamic(self,position,J,mapping): #evolves the activity with J starting from a bump in current position
        #parameters
        N=len(mapping[0])                           
        xi=0.2                    
        maxsteps=50
        a=1.0 #mean activity
        a2=0.1 #sparsity


#initialization
        V=np.zeros(N)
        h=np.zeros(N)
        Vin=np.zeros(N)

        V=self.computeV(mapping,position)
        Vin=V
    
        g=1.1
        th=0
        for step in range(maxsteps):
            h=np.dot(J,V)
            V=np.asarray(list(map(lambda h: self.transfer(h,th,g),h)))
            th=self.fix_parameters(V,h,th,a,a2)
            V=np.asarray(list(map(lambda h: self.transfer(h,th,g),h)))
            g=a/np.mean(V)
            V=g*V
        index=np.where(V==max(V))  
        return mapping[index[0]][0]
        #print("Dynamic step: "+str(step)+" done, mean: "+str(mean(V))+" sparsity: "+str(pow(mean(V),2)/mean(pow(V,2))))

    def attractor_distrib(self,side,J,grid,iterations,subcells):
        ncells=(subcells*subcells)
        unit=side/(2*float(subcells))
        spacing=side/(float(subcells))
        xv=np.linspace(unit,side-unit,subcells)
        yv=np.linspace(unit,side-unit,subcells)
    
        freq=np.zeros((subcells,subcells,ncells))
    
        for j in range(subcells):	     	
            for i in range(subcells):
                for t in range(iterations):
                       loc=np.zeros(2)
                       arrival=np.zeros(2)
                       r=np.zeros(2)
                       loc[0]=np.random.uniform(xv[i]-unit,xv[i]+unit)
                       loc[1]=np.random.uniform(yv[j]-unit,yv[j]+unit)
                       r=self.dynamic(loc,J,grid)
                       arrival[0]=r[0]//spacing
                       arrival[1]=r[1]//spacing
                       arrival=arrival.astype(int)
                       print(arrival)
                       freq[arrival[0]][arrival[1]][i+j]=freq[arrival[0]][arrival[1]][i+j]+1
        meanf=np.zeros((subcells,subcells))
        tot=sum(freq)
        for i in range(subcells):
            for j in range(subcells):
                meanf[i][j]=np.mean(freq[i][j])/tot
                sns.heatmap(meanf, linewidths=.5)        
    
        return meanf




    def random_trajectory(self,L,n,t):#x=storage position vector, t=time steps
        x=np.zeros((t,2))
        #col=arange(t)
        x[0][0]=np.random.uniform(0,L)
        x[0][1]=np.random.uniform(0,L)
        theta=np.zeros(t)
        theta[0]=np.random.uniform(0,2*np.pi)
        
        for i in range(1,t):
            x[i][0]=x[i-1][0]+0.5*(L/np.float(n))*np.cos(theta[i-1])
            x[i][1]=x[i-1][1]+0.5*(L/np.float(n))*np.sin(theta[i-1])
            theta[i]=np.random.normal(theta[i-1],np.pi/2,1)    
            
            if np.logical_or(x[i][0]>L,x[i][0]<0):
                theta[i-1]=np.pi-theta[i-1]
                x[i][0]=x[i-1][0]+0.5*(L/np.float(n))*np.cos(theta[i-1])
            if np.logical_or(x[i][1]>L,x[i][1]<0):
                theta[i-1]=-theta[i-1]
                x[i][1]=x[i-1][1]+0.5*(L/np.float(n))*np.sin(theta[i-1])
        #scatter(x.T[0],x.T[1],c=col,cmap=cm.jet)        
        return x   
            
        #genera traiettoria random con hard bounds e inerzia direzonale
        
        
    def ordered_trajectory(self,L,n,t):
        N=n*n
        x=np.zeros((t,2))
        x[0][0]=np.random.randint(0,N)
        x[0][1]=np.random.randint(0,N)
        while (x[i][0]==n and i<t):
            x[i+1][0]=x[i][0]+np.random.choice([-1,0,1],size=1,p=[1/3,1/3,1/3])
            x[i+1][1]=x[i][1]
            
    def sample_regular_trajectory(self,L,n,epochs): #genera traiettoria random. ciascun neurone viene visitato epochs volte. 
        x=[]
        N=n*n
        pos=np.zeros((N,2))
        for i in range(n):
            for j in range(n):
                pos[n*i+j][0]=i*(L/np.float(n))
                pos[n*i+j][1]=j*(L/np.float(n))
        for e in range(epochs):
            np.random.shuffle(pos)
            x.append(pos)
        return np.asarray(x).reshape((N*epochs,2))

print("ciao")
