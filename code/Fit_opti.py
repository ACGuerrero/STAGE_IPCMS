import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
import matplotlib 
import inspect

dataSi = np.loadtxt("C:/Users/MAISON/Desktop/StageM1_LEROY_CASTILLO/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")
dataair = np.loadtxt("C:/Users/MAISON/Desktop/StageM1_LEROY_CASTILLO/Si substrate/Scanning_200GHZ-1320GHZ_air.txt")

def fit(fct,xdata,ydata,pas):
    j=np.size(xdata);A=np.zeros(j);f=np.zeros(j);phi=np.zeros(j)
    for i in range(j-pas-1):
        P=sc.optimize.curve_fit(fct, xdata[i:pas+i], ydata[i:pas+i])[0]
        A[i]=P[0]
        f[i]=ydata[i+int(pas/2)]
        phi[i]=P[1]
    return A,f,phi

def cos(A,f,phi):
    return np.exp(A)*np.cos(f+phi)

guess=np.array([200,np.pi])
pas=108
xdata=dataSi[:,3]
ydata=dataSi[:,2]
Asi,fsi,phisi=fit(cos,xdata,ydata,pas)



plt.figure()
#plt.plot(xdata,ydata)
plt.plot(xdata,Asi)
#plt.yscale('log')
plt.show()
