import numpy as np 
import matplotlib.pyplot as plt 

def lin(x,xdata,ydata):
    nr=np.size(x)
    y=np.zeros(nr)
    dx=xdata[1]-xdata[0]
    index=np.int32(x/dx)
    for i in range(nr):
        idx=index[i]
        slope=(ydata[idx+1]-ydata[idx])/(xdata[idx+1]-xdata[idx])
        y[i]=slope*(x[i]-xdata[idx])+ydata[idx]
    return y

def sin(x):
    return np.sin(x)
N=100
x=np.linspace(0,20,num=N)
xr=np.zeros(N)
for i in range(N):
    xr[i]=x[i]+np.random.rand()*0.5-np.random.rand()*0.5

plt.figure()
plt.scatter(x,sin(x),s=5)
plt.scatter(x,sin(xr),s=5)
plt.show()

y=lin(x,xr,sin(xr))
plt.figure()
plt.plot(x,y)
plt.show()

