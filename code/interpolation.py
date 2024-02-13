import numpy as np
import matplotlib.pyplot as plt

# Linear interpolation function

def linear_interpolation(xgiven, ygiven, xinter):
    N = np.size(xinter)
    yinter = np.zeros(N)

    for k in range(N):
        i = np.argmin(np.abs(xgiven - xinter[k]))

        # Ensure i+1 doesn't go beyond the array size
        if i < np.size(xgiven) - 1:
            yinter[k] = ygiven[i] + (xinter[k] - xgiven[i]) * (ygiven[i+1] - ygiven[i]) / (xgiven[i+1] - xgiven[i])
        else:
            yinter[k] = ygiven[i]  # Use the last available value for extrapolation

    return yinter

if __name__ == '__main__':
    # We want to calculate the difference between these two sets of data

    data_air = np.loadtxt('Scanning_900GHZ-1200GHZ_air_enveloppe.txt')
    data_NiO = np.loadtxt('Scanning_900GHZ-1200GHZ_NIO_enveloppe.txt')
    xair = data_air[:,0]
    yair = data_air[:,1]
    xNiO = data_NiO[:,0]
    yNiO = data_NiO[:,1]

    # However, the x-values are not the same, nor they have the same number of points, so we need to interpolate the data

    xinter = np.linspace(900,1200,1000)
    yair_inter = linear_interpolation(xair,yair,xinter)
    yNiO_inter = linear_interpolation(xNiO,yNiO,xinter)

    # We plot the interpolated data

    plt.figure()
    plt.plot(xinter,yair_inter,label='Air')
    plt.plot(xinter,yNiO_inter,label='NiO')
    plt.legend()
    plt.savefig('interpolated_data.pdf')
    plt.close()

    # We now plot the difference
    plt.figure()
    plt.plot(xinter,yair_inter/yNiO_inter)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Difference (nA)')
    plt.title('Difference between Air and NiO')
    plt.yscale('log')
    plt.savefig('difference.pdf')
    plt.close()
