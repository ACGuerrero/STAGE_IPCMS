import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 

def fit(fct, xdata, ydata, interval_size, P0, step_size):

    N = np.size(xdata)
    freq = np.zeros(N)

    A = np.zeros(N)
    phi = np.zeros(N)
    count = 0

    for i in range(0, N-interval_size-1, step_size):
        P = sc.optimize.curve_fit(fct, xdata[i:i+interval_size], ydata[i:i+interval_size], P0, maxfev=5000)[0]
        if P[0] > 0.01 :
            A[count] = P[0]
            phi[count] = P[1]
            freq[count] = xdata[i+int(interval_size/2)]
            count += 1
    print(f'\nCurve fitted, input size is {N}, output size is {count}')
    return  np.array(freq)[:count], np.array(A)[:count], np.array(phi)[:count]

def fit1(fct, xdata, ydata, interval_size, P0, step_size):

    N = np.size(xdata)
    new_xvalues = np.zeros(N)
    parameters = np.zeros(N,len(P0))
    count = 0

    for i in range(0, N-interval_size-1, step_size):
        P = sc.optimize.curve_fit(fct, xdata[i:i+interval_size], ydata[i:i+interval_size], P0, maxfev=5000)[0]
        if np.exp(P[0]) > 0.01 :
            A[count] = np.exp(P[0])
            phi[count] = P[1]
            freq[count] = xdata[i+int(interval_size/2)]
            count += 1
    print(f'\nCurve fitted, input size is {N}, output size is {count}')
    return  np.array(freq)[:count], np.array(A)[:count], np.array(phi)[:count]



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
    freq, AA, phi = fit(cos, xdata, ydata, interval_size, P0, step_size)

    # PLOTTING RESULTS
    plt.figure()
    plt.plot(freq, AA, color = 'r')
    plt.plot(xdata,ydata, alpha=0.2)
    plt.show()
