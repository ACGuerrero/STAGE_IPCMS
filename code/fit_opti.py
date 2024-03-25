import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 

def fit(fct, xdata, ydata, interval_size, P0, step_size, maxfev=10000):

    N = np.size(xdata)
    new_xvalues = np.zeros(N)
    parameters = np.zeros((N, len(P0)))
    count = 0

    for i in range(0, N-interval_size-1, step_size):
        P = sc.optimize.curve_fit(fct, xdata[i:i+interval_size], ydata[i:i+interval_size], P0, maxfev)[0]
        if P[0] > 0.01 :
            parameters[count] = P
            new_xvalues[count] = xdata[i+int(interval_size/2)]
            count += 1
            P0 = P
    print(f'\nCurve fitted, input size is {N}, output size is {count}')
    return  new_xvalues[:count], parameters[:count].T


def Atrans(f, R1, R2, n):
    l = 285e-6 # Keeping l as a floating-point number
    c = 3e8    # Keeping c as a floating-point number
    return (1-R1)*(1-R2)/((1-np.sqrt(R1*R2))**2+4*np.sqrt(R1*R2)*np.sin(2*np.pi*l*n/c*f)**2)  

def cos(f, A, phi):
    return A*np.cos(2.358*f+phi)


if __name__ == '__main__':
    # LOAD DATA
    dataSi = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")

    # EXTRACTING DATA
    xdata = dataSi[:,3]
    ydata = dataSi[:,2]

    # FUNCTION PARAMETERS
    interval_size=108
    P0=[np.log(10), 0]
    step_size = 3

    # FUNCTION CALLING
    freqs, params = fit(cos, xdata, ydata, interval_size, P0, step_size)
    # PLOTTING RESULTS
    plt.figure()
    plt.plot(freqs, params[0])
    plt.show()

