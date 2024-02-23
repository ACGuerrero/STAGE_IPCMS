import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sc 

D=np.zeros((6,6))
#PARIS 1, NANCY 2, STRASBOURG 3, MULHOUSE 4, BESANCON 5, DIJON 6
D[0,1]=312;D[0,2]=445;D[0,3]=440;D[0,4]=365;D[0,5]=290
D[1,2]=133;D[1,3]=161;D[1,4]=184;D[1,5]=191
D[2,3]=102;D[2,4]=227;D[2,5]=282
D[3,4]=130;D[3,5]=204
D[4,5]=83
DD=np.transpose(D)+D
T=np.arange(1,7,dtype=int)-1
TT=np.zeros((2,6),dtype=int) #Tableau valeur ordre 
TT[0,:]=T
TT[1,:]=T


def permutation(TT,j,k):
    a=0;b=0
    a=TT[0,j];b=TT[0,k];TT[0,j]=b;TT[0,k]=a
    return TT

def KM(TT):
    DT=np.zeros(6,dtype=int)
    KM=0
    for i in range(5):
        DT[i]=DD[TT[0,i],TT[0,i+1]]
        KM=DT[i]+KM
    return KM

compare=np.zeros((6,6))

for i in range(2,6):
    compare[0,0]=KM(TT)    
    TT=permutation(TT,i-1,i)
    print(TT)
    compare[i,0]=KM(permutation(TT,i-1,i))

print(compare)
