import numpy as np
#import scipy.special  
import seaborn as sns
import matplotlib.pyplot as plt
#from yapf.yapflib.yapf_api import FormatCode
from numpy import linalg as LA
from functions import functions as fc

class Jlearn:

#__init__(self,n)
#ambient(self,l)
#computeV(self,currentposition)
#strenght(self,currentposition,eta)
#learn(self,path,eta,A)
#zerostrenght(self)
#hebbstrenght(self)
#random_trajectory(self,t)
#ordered_trajectory(self,t)
#sample_regular_trajectory(self,epochs)
#J_txt(A,l,eta,epochs)
#grid_txt(B,l)

    
    def __init__(self,n):
        self.n=n #linear dimension
        
        self.N=n*n #number of neurons
        self.L=100 #lenght of the grid
        self.sigma=1*np.float(self.L/n) 
        self.c=1*np.float(self.L/n)*5 #cutoff
        self.grid=np.zeros((pow(n,2),2)) #firing places' grid
        self.J=np.zeros((n*n,n*n)) #connettivity matrix
        self.hebbJ=np.zeros((n*n,n*n)) #hebbian CM 
        self.V=np.zeros(n*n) #activity of the network
        self.func=fc(self.sigma,self.c)
    
    def ambient(self,l):#n=square root of number of neurons, l=randomness ,L dimension
        for i in range(self.n):
            for j in range(self.n):
                self.grid[self.n*i+j][0]=np.random.uniform(np.float(self.L)/np.float(self.n)*(i),np.float(self.L)/np.float(self.n)*(i+l))
                self.grid[self.n*i+j][1]=np.random.uniform(np.float(self.L)/np.float(self.n)*(j),np.float(self.L)/np.float(self.n)*(j+l))
        return         
    
    def computeV(self,currentposition): #computes the activity of the network depending on the current position of the external input
        for i in range(self.N):
                self.V[i]=self.func.kernel(self.grid[i],currentposition)
    
    def strenght(self,currentposition,eta): #upgrades CM with a learning rate eta dep. on the activity, fills J
        deltaJ=np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                if j!=i:
                    deltaJ[i][j]=eta*(self.V[i])*(self.V[j]) 
                else:
                    deltaJ[i][j]=0
                    
        for i in range(self.N):
            for j in range(self.N):
                self.J[i][j]=self.J[i][j]+deltaJ[i][j]
        #a=sum(self.J)
        #self.J=self.J/a
        self.J=self.J/LA.norm(self.J)
            
    def learn(self,path,eta,A): #total learning of the CM in t steps of a path
        et=eta
        t=path.shape[0]
        m=np.zeros(t)
        for s in range(t):
            self.computeV(path[s]) #compute V on the current position of path's
            self.strenght(path[s],et) #reinforcement of CM
            m[s]=self.func.overlap(self.J,A) #evaluates overlap between J and A
            if s%100==0:
                print(s)
        plt.plot(np.arange(t),m) #plots overlap over t
        plt.xlabel('t')
        plt.ylabel('overlap')
        plt.show()
        return m
           
            
    
    def zerostrenght(self): #generates a starting randm uniform CM
        for i in range(self.N):
            for j in range(self.N):
                self.J[i][j]=np.random.randint(0,10,size=1)
        #a=np.sum(self.J)
        #self.J=self.J/a
        self.J=self.J/LA.norm(self.J)
       
        
    def hebbstrenght(self):
        #generates an hebbian CM
        for i in range(self.N):
            for j in range(self.N):
                x=self.grid[i]
                y=self.grid[j]
                self.hebbJ[i][j]=self.func.kernel(x,y)
        #a=sum(self.hebbJ)
        #self.hebbJ=self.hebbJ/a
        self.hebbJ=self.hebbJ/LA.norm(self.hebbJ)
        

#    def random_trajectory(self,t):#x=storage position vector, t=time steps
#        x=np.zeros((t,2))
#        #col=arange(t)
#        x[0][0]=np.random.uniform(0,self.L)
#        x[0][1]=np.random.uniform(0,self.L)
#        theta=np.zeros(t)
#        theta[0]=np.random.uniform(0,2*np.pi)
#        
#        for i in range(1,t):
#            x[i][0]=x[i-1][0]+0.5*(self.L/np.float(self.n))*np.cos(theta[i-1])
#            x[i][1]=x[i-1][1]+0.5*(self.L/np.float(self.n))*np.sin(theta[i-1])
#            theta[i]=np.random.normal(theta[i-1],np.pi/2,1)    
#            
#            if np.logical_or(x[i][0]>self.L,x[i][0]<0):
#                theta[i-1]=np.pi-theta[i-1]
#                x[i][0]=x[i-1][0]+0.5*(self.L/np.float(self.n))*np.cos(theta[i-1])
#            if np.logical_or(x[i][1]>self.L,x[i][1]<0):
#                theta[i-1]=-theta[i-1]
#                x[i][1]=x[i-1][1]+0.5*(self.L/np.float(self.n))*np.sin(theta[i-1])
#        #scatter(x.T[0],x.T[1],c=col,cmap=cm.jet)        
#        return x   
#            
#        #genera traiettoria random con hard bounds e inerzia direzonale
#        
#        
#    def ordered_trajectory(self,t):
#        x=np.zeros((t,2))
#        x[0][0]=np.random.randint(0,self.N)
#        x[0][1]=np.random.randint(0,self.N)
#        while (x[i][0]==self.n and i<t):
#            x[i+1][0]=x[i][0]+np.random.choice([-1,0,1],size=1,p=[1/3,1/3,1/3])
#            x[i+1][1]=x[i][1]
#        return x
#            
#    def sample_regular_trajectory(self,epochs): #genera traiettoria random. ciascun neurone viene visitato epochs volte. 
#        x=[]
#        pos=np.zeros((self.N,2))
#        for i in range(self.n):
#            for j in range(self.n):
#                pos[self.n*i+j][0]=i*(self.L/np.float(self.n))
#                pos[self.n*i+j][1]=j*(self.L/np.float(self.n))
#        for e in range(epochs):
#            np.random.shuffle(pos)
#            x.append(pos)
#        return np.asarray(x).reshape((self.N*epochs,2))

    
    def J_txt(self,l,eta,epochs): #prints J in a .txt reporting the trajectory,learning rate and epochs of the learning
        B=np.matrix(self.J)
        name_of_file="J("+np.str(l)+","+np.str(eta)+","+np.str(epochs)+")"
        file=open(name_of_file + ".txt","w+")
        for line in B:
            np.savetxt(file,line,fmt='%.2f')
        file.close()
    
    def grid_txt(self,l): #prints grid in a .txt reporting the l parameter for randomness
        A=np.matrix(self.grid)
        name_of_file="grid("+np.str(len(self.grid))+","+np.str(l)+")"
        file=open(name_of_file + ".txt","w+")
        for line in A:
            np.savetxt(file,line,fmt='%.2f')
        file.close()            

print("ciao")
