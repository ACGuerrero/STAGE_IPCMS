import numpy as np
import matplotlib.pyplot as plt
from fit_opti import *

if __name__ == '__main__':
    # LOAD DATA
    dataSi = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")
    dataair = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_air.txt")

    #EXTRACT DATA
    f_si=dataSi[:,3]
    A_si=dataSi[:,2]
    f_air=dataair[:,3]
    A_air=dataair[:,2]/2

    # FUNCTION PARAMETERS
    interval_size = 128
    step_size = 4
    P0_air=[A_air[0], 0]
    P0_si=[A_si[0], 0]

    # FITTING
    freqs_si, params_si = fit(cos, f_si, A_si, interval_size, P0_si, step_size)
    freqs_air, params_air = fit(cos, f_air, A_air, interval_size, P0_air, step_size)

    # INTERPOLATION
    params_interp = np.interp(freqs_air,freqs_si,params_si[0])

    # RATIO
    ratio=params_air[0]/params_interp  # ratio calculation
    P0_ratio=[0.3,0.3,3.1] # guess
    freq_fit_ratio, params_fit_ratio = fit(Atrans,freqs_air, ratio, interval_size, P0_ratio, step_size) # fitting

    P=sc.optimize.curve_fit(Atrans,freqs_air, ratio, P0_ratio, maxfev = 10000)[0]

    yatrans = np.zeros_like(freq_fit_ratio)

    for i in range(len(freq_fit_ratio)):
        yatrans[i] = Atrans(freq_fit_ratio[i],
                            params_fit_ratio[0][i],
                            params_fit_ratio[1][i],
                            params_fit_ratio[2][i]) 
    

    # PLOTTING

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # Create subplots with 1 row and 2 columns
    # Plot Si data and fit
    axs[0].plot(freqs_si, params_si[0], color='black', label='Si Fit')
    axs[0].plot(f_si, A_si, alpha=0.5)
    axs[0].set_title('Si Data')

    # Plot Air data and fit
    axs[1].plot(freqs_air, params_air[0], color='black', label='Air Fit')
    axs[1].plot(f_air, A_air, alpha=0.5)
    axs[1].set_title('Air Data')

    # Plot Ratio

    axs[2].plot(freqs_air, ratio, label='Ratio', alpha=0.5)
    axs[2].plot(freq_fit_ratio,  yatrans, color = 'black')
    axs[2].plot(freq_fit_ratio, Atrans(freq_fit_ratio, P[0],P[1],P[2]), color = 'red')
    axs[2].set_title('Ratio')

    # Add legend and show plot
    plt.legend()
    plt.show()