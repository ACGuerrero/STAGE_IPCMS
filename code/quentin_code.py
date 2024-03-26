import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.optimize import *
import timeit
from mpl_toolkits import mplot3d
from matplotlib import cm
from scipy import signal
import scipy



dir="C:/Users/quent/OneDrive/Bureau/THz/test_20220530/"

dir2="C:/Users/quent/OneDrive/Bureau/THz/test_20220506/"

dir3="C:/Users/quent/OneDrive/Bureau/THz/test_20220505/"

dir4="C:/Users/quent/OneDrive/Bureau/THz/test_20220503/"

dir5="C:/Users/quent/OneDrive/Bureau/THz/test_20220610/"

dir6="C:/Users/quent/OneDrive/Bureau/THz/NiOpeak2/"

dir7="C:/Users/quent/OneDrive/Bureau/THz/NiO_B/"

air7="Air_50MHz_50ms.txt"

air7950="Air_50MHz_50ms_950-1100GHz.txt"

NiO1="NiO_50MHz_50ms.txt"

NiO950="NiO_50MHz_50ms_950-1100GHz.txt"

NiOB="NiO_Bfield_50MHz_50ms.txt"

NiOB950="NiO_Bfield_50MHz_50ms_950-1100GHz.txt"

filename="air_step50MHz_int100ms.txt"

filename1="air_envelope_step50MHz_int100ms_24points.txt"

filename2="Si_step50MHz_int100ms.txt"

air2="air_100ms_data2.txt"

Si2="silicium_350µm_100ms_data.txt"

saphir="saphir_500µm_100ms_data.txt"

verre="verre_400µm_100ms_data.txt"

file="empty_and_lactose_raw.txt"

air900="air_step30MHz_int200ms.txt"

airn="air_step50MHz_int100ms.txt"

NiO="NiO_step50MHz_int100ms.txt"

NiO900="NiO_step30MHz_int200ms.txt"

NiO2="NiO_step50MHz_int500ms.txt"

NiO1120="NiO_step20MHz_int500ms.txt"

air1100="air_step50MHz_int500ms.txt"

air1120="air_step20MHz_int500ms.txt"


##Frequency analysis

def open1(dir, plot):
    file=open(dir)
    L=file.readlines()
    f=[]
    C=[]
    f2=[]
    C2=[]
    for i in range(1,24396):
        a=L[i].split()
        f.append(float(a[2]))
        C.append(float(a[1]))
    for i in range(24396, 48791):
        a=L[i].split()
        f2.append(float(a[2]))
        C2.append(float(a[1]))
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(np.array(f)/10**3,C)
        plt.plot(np.array(f2)/10**3, C2)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
        plt.figure()
        ax=plt.subplot(211)
        yf=scipy.fft.ifft(C)
        C2.reverse()
        yf2=scipy.fft.ifft(C2)
        xf=np.arange(0, len(f)/(2.64*10**12), 1/(2.64*10**12))
        xf2=np.arange(0, len(f2)/(2.64*10**12), 1/(2.64*10**12))
        plt.plot(xf[0:len(xf)], np.abs(yf), label="air")
        plt.plot(xf2[0:len(xf2)],np.abs(yf2), label="lactose")
        plt.legend()
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(xf[0:len(xf)], np.angle(yf))
        plt.plot(xf2[0:len(xf2)],np.angle(yf2))
        plt.grid()
        plt.show()
    file.close()
    return(f, C, f2, C2)



def openfile(dir, plot):
    file=open(dir)
    L=file.readlines()
    f=[]
    C=[]
    for i in range(1,len(L)):
        a=L[i].split()
        f.append(float(a[3]))
        C.append(float(a[2]))
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(np.array(f)/10**3,C)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    file.close()
    return(f, C)


"""def ampl(dir, plot):
    f,C=openfile(dir, 0)

    plt.figure()
    for j in range (54, 55):
        i=0
        ampl=[]
        fampl=[]
        while i<=len(C):
            a=np.max(C[i:i+j])-np.min(C[i:i+j])
            ia=f[int((np.argmax(C[i:i+j])+np.argmin(C[i:i+j]))/2)+i]
            ampl.append(a)
            fampl.append(ia)
            i=i+j
        if plot==1:
            plt.semilogy(np.array(fampl)/10**3,ampl, label=j)
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Photocurrent (nA)")
    plt.legend()
    plt.grid()
    plt.show()
    return(fampl,ampl)


def openenv(dir, plot):
    file=open(dir)
    L=file.readlines()
    f=[]
    C=[]
    for i in range(1,len(L)):
        a=L[i].split()
        f.append(float(a[0]))
        C.append(float(a[1]))
    if plot==1:
        plt.figure()
        plt.grid()
        plt.semilogy(np.array(f)/10**3,C)
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    file.close()
    return(f, C)


def func_fit(f, A, a, phi, om, f0):
    return A*np.exp(-a*(f-f0))*np.cos(om*(f-f0)+phi)

def fit(dir, plot):
    f,C=openfile(dir, 0)
    f=np.array(f)
    P0=[0,0.001,0,2*np.pi/3, 100]
    P=curve_fit(func_fit, f, C, P0, maxfev=10000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(f, func_fit(f, *P), label="fit")
        plt.plot(f, C, ".", label="Data")
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.legend()
        plt.show()
    return P """


def cos(f, A, phi):
    return np.exp(A)*np.cos(2.358*f+phi)

# Fitting function that uses the openfile function and fits a cosine function to the data
def cosfit(dir, plot):
    f, C = openfile(dir, 0)
    f = np.array(f)
    A = []
    freq = []
    phi = []
    j = 108  # parameter for the range of data to fit
    if plot == 1:
        plt.figure()
        plt.grid()
        plt.plot(f / 10 ** 3, C, ".", label="Data")
    for i in range(0, len(f) - j - 1, 3):
        P0 = [10, 0]  # Initial parameters for curve fitting
        # Curve fitting using scipy's curve_fit function
        P = curve_fit(cos, f[i:i + j], C[i:i + j], P0, bounds=([-np.inf, -np.inf], [np.inf, np.inf]), maxfev=5000)[0]
        if np.exp(P[0]) > 0.01:  # Check if the fit is significant
            A.append(np.exp(P[0]))
            freq.append(f[i + int(j / 2)])
            phi.append(P[1])
        if plot == 1:
            plt.plot(f[i:i + j] / 10 ** 3, cos(f[i:i + j], *P), "r")  # Plot the fitted curve
    if plot == 1:
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    return A, phi, freq
  
# This code 
# Another fitting function with variations in parameters and conditions
def cosfit2(dir, plot):
    f, C = openfile(dir, 0)
    f = np.array(f)
    A = []
    freq = []
    phi = []
    j = 27  # parameter for the range of data to fit
    if plot == 1:
        plt.figure()
        plt.grid()
        plt.plot(f, C, ".", label="Data")
    for i in range(0, len(f) - j - 1, 3):
        if i < (len(f) - j - 1) / 3:
            P0 = [10, 0]  # Initial parameters for curve fitting
        else:
            P0 = [2, 0]  # Different initial parameters for the second part of the data
        # Curve fitting using scipy's curve_fit function
        P = curve_fit(cos, f[i:i + j], C[i:i + j], P0, maxfev=5000)[0]
        if np.exp(P[0]) > 0.001:  # Check if the fit is significant
            A.append(np.exp(P[0]))
            freq.append(f[i + int(j / 2)])
            phi.append(P[1])
        if plot == 1:
            plt.plot(f[i:i + j], cos(f[i:i + j], *P), "r")  # Plot the fitted curve
    if plot == 1:
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    return A, phi, freq

def cosfit3(dir, plot):
    f, C, f2, C2 = open1(dir, 0)
    f=np.array(f)
    f2=np.array(f2)
    A=[]
    A2=[]
    freq=[]
    freq2=[]
    phi=[]
    phi2=[]
    j=108
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(f, C, ".", label="Data")
    for i in range(0, len(f)-j-1, 3):
        if i<(len(f)-j-1)/3:
            P0=[10, 0]
            P1=[-1, 0]
            P=curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
            P2=curve_fit(cos, f2[i:i+j], C2[i:i+j], P1, maxfev=5000)[0]
        elif i>(len(f)-j-1)/3 and i<2*(len(f)-j-1)/3:
            P0=[1, 0]
            P=curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
            P2=curve_fit(cos, f2[i:i+j], C2[i:i+j], P0, maxfev=5000)[0]
        else:
            P0=[1, 0]
            P1=[10,0]
            P=curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
            P2=curve_fit(cos, f2[i:i+j], C2[i:i+j], P1, maxfev=5000)[0]
        if np.exp(P[0])>0.005:
            A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[1])
        if np.exp(P2[0])>0.005:
            A2.append(np.exp(P2[0]))
            freq2.append(f2[i+int(j/2)])
            phi2.append(P2[1])
        if plot==1:
            plt.plot(f[i:i+j], cos(f[i:i+j], *P), "r")
            plt.plot(f2[i:i+j], cos(f2[i:i+j], *P2), "y")
    if plot==1:
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    return (A, phi, freq, A2, phi2, freq2)

def cosfit4(dir, plot):
    f, C = openfile(dir, 0)
    f=np.array(f)
    A=[]
    freq=[]
    phi=[]
    j=80
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(f, C, ".", label="Data")
    for i in range(0, len(f)-j-1, 3):
        P0=[1, 0]
        P=curve_fit(cos, f[i:i+j], C[i:i+j], P0, bounds=([-np.inf, -np.inf],[np.inf,np.inf]), maxfev=5000)[0]
        if np.exp(P[0])>0.01:
            A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[1])
        if plot==1:
            plt.plot(f[i:i+j], cos(f[i:i+j], *P), "r")
    if plot==1:
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    return (A, phi, freq)

def cosfit5(dir, plot):
    f, C = openfile(dir, 0)
    f=np.array(f)
    A=[]
    freq=[]
    phi=[]
    j=133
    if plot==1:
        plt.figure()
        plt.grid()
        plt.plot(f, C, ".", label="Data")
    for i in range(0, len(f)-j-1, 3):
        P0=[1, 0]
        P=curve_fit(cos, f[i:i+j], C[i:i+j], P0, bounds=([-np.inf, -np.inf],[np.inf,np.inf]), maxfev=5000)[0]
        if np.exp(P[0])>0.01:
            A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[1])
        if plot==1:
            plt.plot(f[i:i+j], cos(f[i:i+j], *P), "r")
    if plot==1:
        plt.legend()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Photocurrent (nA)")
        plt.show()
    return (A, phi, freq)

def cosamplphase2(dir):
    A, phi, f, A2, phi2, f2=cosfit3(dir, 0)
    A=np.abs(A)
    A2=np.abs(A2)
    phi=np.array(phi)
    phi2=np.array(phi2)
    f=np.array(f)
    f2=np.array(f2)
    phi=phi%(2*np.pi)-np.pi
    phi2=phi2%(2*np.pi)-np.pi
    plt.figure()
    ax=plt.subplot(211)
    plt.semilogy(f/10**3, A, label="air")
    plt.semilogy(f2/10**3, A2, label="lactose")
    plt.legend()
    plt.grid()
    plt.ylabel("Photocurrent (nA)")

    ax=plt.subplot(212)
    plt.plot(f/10**3, phi, label="air")
    plt.plot(f2/10**3, phi2, label="lactose")
    plt.grid()
    plt.ylabel("Phase (rad)")
    plt.xlabel("Frequency (THz)")
    plt.legend()
    plt.show()


def dividebyair3(dir, plot):
    A1, phi1, f1, A2, phi2, f2=cosfit3(dir, 0)
    phi1=np.array(phi1)%(2*np.pi)-np.pi
    phi2=np.array(phi2)%(2*np.pi)-np.pi
    T=[]
    deltaphi=[]
    f=np.linspace(100, 1320, 8097)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1
        if i != 100 and i != 1320:
            A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
            Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
            p1.append((phi1[k]+phi1[k+1])/2)
            p2.append((phi2[r]+phi2[r+1])/2)
    A=np.array(A)
    p1=np.array(p1)
    Al=np.array(Al)
    p2=np.array(p2)
    T=Al/A
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    f=list(f)
    f.pop(0)
    f.pop(len(f)-1)
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(len(T)):
        if T[i]>1.1 or T[i]<0 or deltaphi[i]>deltaphi[i-1]+0.3 or deltaphi[i]<deltaphi[i-1]-0.3:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        """plt.legend()"""
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[10:len(f)])/10**3, deltaphi[10:len(f)])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)


def cosamplphase(dir):
    A, phi, f=cosfit(dir, 0)
    A=np.abs(A)
    phi=np.array(phi)
    phi=phi%(2*np.pi)-np.pi
    plt.figure()
    ax=plt.subplot(211)
    plt.semilogy(np.array(f)/10**3, A)
    plt.grid()
    plt.ylabel("Photocurrent (nA)")

    ax=plt.subplot(212)
    plt.plot(np.array(f)/10**3, phi)
    plt.grid()
    plt.ylabel("Phase (rad)")
    plt.xlabel("Frequency (THz)")
    plt.show()




def dividebyair(dir1, plot):
    A1, phi1, f1=cosfit(dir+filename, 0)
    A2, phi2, f2=cosfit(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(100, 1317, 8097)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.05 or deltaphi[i]<deltaphi[i-1]-0.05 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)




def dividebyair2(dir1, plot):
    A1, phi1, f1=cosfit2(dir2+air2, 0)
    A2, phi2, f2=cosfit2(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(100, 1297, 3991)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    phi1=np.array(p1)
    phi2=np.array(p2)
    deltaphi=(phi2-phi1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.3 or deltaphi[i]<deltaphi[i-1]-0.3:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)

def dividebyair4(dir1, plot): #NiO
    A1, phi1, f1=cosfit(dir5+airn, 0)
    A2, phi2, f2=cosfit(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(100, 1317, 8097)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)



def dividebyair5(dir1, plot): #NiO900
    A1, phi1, f1=cosfit4(dir5+air900, 0)
    A2, phi2, f2=cosfit4(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(900, 1200, 9000)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)


def dividebyair6(dir1, plot): #NiO1/NiOB
    A1, phi1, f1=cosfit(dir7+air7, 0)
    A2, phi2, f2=cosfit(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(800, 1300, 9000)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)



def dividebyair7(dir1, plot): #NiO950/NiOB950
    A1, phi1, f1=cosfit(dir7+air7950, 0)
    A2, phi2, f2=cosfit(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(950, 1100, 3000)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        """plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])"""
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)


"""def dividebyair6(dir1, plot): #NiO2
    A1, phi1, f1=cosfit(dir6+air1100, 0)
    A2, phi2, f2=cosfit(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(1100, 1200, 1500)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)


def dividebyair7(dir1, plot): #NiO1120
    A1, phi1, f1=cosfit5(dir6+air1120, 0)
    A2, phi2, f2=cosfit5(dir1, 0)
    T=[]
    deltaphi=[]
    f=np.linspace(1120, 1150, 1500)
    A=[]
    Al=[]
    p1=[]
    p2=[]
    for i in f:
        k=0
        r=0
        while ((f1[k]>=i and f1[k+1]>=i) or (f1[k]<=i and f1[k+1]<=i)) and k<(len(f1)-2):
            k+=1


        while ((f2[r]>=i and f2[r+1]>=i) or (f2[r]<=i and f2[r+1]<=i)) and r<(len(f2)-2):
            r+=1

        A.append((A1[k]-A1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+A1[k])
        Al.append((A2[r]-A2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+A2[r])
        p1.append((phi1[k]-phi1[k+1])/(f1[k]-f1[k+1])*(i-f1[k])+phi1[k])
        p2.append((phi2[r]-phi2[r+1])/(f2[r]-f2[r+1])*(i-f2[r])+phi2[r])
    A=np.array(A)
    Al=np.array(Al)
    T=Al/A
    p1=np.array(p1)
    p2=np.array(p2)
    deltaphi=(p2-p1)%(2*np.pi)-np.pi
    j=0
    I=[]
    T=list(T)
    f=list(f)
    deltaphi=list(deltaphi)
    A=list(A)
    Al=list(Al)
    for i in range(1, len(T)):
        if T[i]>1.1 or deltaphi[i]>deltaphi[i-1]+0.2 or deltaphi[i]<deltaphi[i-1]-0.2 or T[i]<0:
            j+=1
            I.append(i)
    for i in range(j):
        T.pop(I[i]-i)
        f.pop(I[i]-i)
        deltaphi.pop(I[i]-i)
        A.pop(I[i]-i)
        Al.pop(I[i]-i)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(np.array(f[3:len(f)])/10**3, T[3:len(f)])
        plt.semilogy(np.array(f[3:len(f)])/10**3, A[3:], label="Air")
        plt.semilogy(np.array(f[3:len(f)])/10**3, Al[3:])
        plt.legend()
        plt.grid()
        plt.ylabel("Transmission")

        ax=plt.subplot(212)
        plt.plot(np.array(f[1:len(f)])/10**3, deltaphi[1:len(f)])
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T, deltaphi, f)"""




l=280*10**(-6) #m
c=3*10**8 #m/s

def FP_trans(f, R1, R2, n):
    return (1-R1)*(1-R2)/((1-np.sqrt(R1*R2))**2+4*np.sqrt(R1*R2)*np.sin(2*np.pi*l*n/c*f)**2)

def FP_refl(f, R1, n):
    return ((np.sqrt(R1)-np.sqrt(R1))**2+4*np.sqrt(R1*R1)*np.sin(2*np.pi*l*n/c*f)**2)/((1-np.sqrt(R1*R1))**2+4*np.sqrt(R1*R1)*np.sin(2*np.pi*l*n/c*f)**2)

def FPfit(dir, plot):
    T, deltaphi, f=dividebyair(dir, 0)
    f=np.array(f)*10**9
    P0=[0.3, 0.3, 3.4]
    P=curve_fit(FP_trans, f, T, P0, maxfev=10000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.plot(f[1:]/10**12, T[1:], label="Data")
        plt.plot(f/10**12, FP_trans(f, *P), label="Fit transmission")
        plt.plot()
        plt.grid()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmission")
        plt.legend()
        plt.show()
    return f, P



def FPfit2(dir, plot):
    T, deltaphi, f=dividebyair2(dir, 0)
    f=np.array(f)*10**9
    P0=[0.16, 2.77]
    P=curve_fit(FP_trans, f[1:], T[1:], P0, maxfev=10000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.plot(f[1:]/10**12, T[1:], label="Data")
        plt.plot(f/10**12, FP_trans(f, *P), label="Fit transmission")
        plt.plot(f/10**12, FP_refl(f, *P), label="Reflection")
        plt.grid()
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Transmission")
        plt.legend()
        plt.show()
    return f, P


def divideFP(dir, plot):
    T, deltaphi, f=dividebyair(dir, 0)
    f=np.array(f)*10**9
    P=FPfit(dir, 0)
    T2=FP_trans(f, *P)
    T_intr=T/T2
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**12, T_intr[1::])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1:]/10**12, deltaphi[1:])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T_intr, deltaphi, f/10**12)



def divideFP2(dir, plot):
    T, deltaphi, f=dividebyair2(dir, 0)
    f=np.array(f)*10**9
    P=FPfit2(dir, 0)
    T2=FP_trans(f, *P)
    T_intr=T/T2
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**12, T_intr[1::])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1::]/10**12, deltaphi[1:])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (T_intr, deltaphi, f/10**12)

def deroulphase(dir, plot):
    T, dphi, f = dividebyair(dir, 0)
    j=0
    fi=-1000
    for i in range(2, len(dphi)):
        if abs(dphi[i])>abs(dphi[i-1])-0.05 and  abs(dphi[i])<abs(dphi[i-1])+0.05 and dphi[i]<dphi[i-1]-6 and dphi[i-1]>np.pi-0.05 and f[i]>fi+70:
            j+=1
            fi=f[i]
            print(fi)
        dphi[i-2]=dphi[i-2]+2*j*np.pi
        if dphi[i-2]>2*j*np.pi+np.pi-0.35 and j>0 and f[i]<fi+70:
            dphi[i-2]=dphi[i-2]-2*np.pi
    f=np.array(f)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**3, T[1::])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1:len(f)-2]/10**3, dphi[1:len(f)-2])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return T, dphi, f


def deroulphase4(dir, plot):
    T, dphi, f = dividebyair5(dir, 0)
    j=0
    fi=-1000
    for i in range(2, len(dphi)):
        if abs(dphi[i])>abs(dphi[i-1])-0.05 and  abs(dphi[i])<abs(dphi[i-1])+0.05 and dphi[i]<dphi[i-1]-6 and dphi[i-1]>np.pi-0.05 and f[i]>fi+70:
            j+=1
            fi=f[i]
            print(fi)
        dphi[i-2]=dphi[i-2]+2*j*np.pi
        if dphi[i-2]>2*j*np.pi+np.pi-0.35 and j>0 and f[i]<fi+70:
            dphi[i-2]=dphi[i-2]-2*np.pi
    f=np.array(f)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**3, T[1::])
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1:len(f)-2]/10**3, dphi[1:len(f)-2])
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return T, dphi, f

def deroulphase2(dir, plot):
    T, dphi, f = dividebyair2(dir, 0)
    j=0
    fi=-1000
    for i in range(2, len(dphi)):
        if abs(dphi[i])>abs(dphi[i-1])-0.1 and  abs(dphi[i])<abs(dphi[i-1])+0.1 and dphi[i]<dphi[i-1]-5.8 and dphi[i-1]>np.pi-0.05 and f[i]>fi+70:
            j+=1
            fi=f[i]
            print(fi)
        dphi[i-2]=dphi[i-2]+2*j*np.pi
        if dphi[i-2]>2*j*np.pi+np.pi-0.35 and j>0 and f[i]<fi+70:
            dphi[i-2]=dphi[i-2]-2*np.pi
    f=np.array(f)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**3, T[1::])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1:len(f)-2]/10**3, dphi[1:len(f)-2])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return T, dphi, f

def deroulphase3(dir, plot):
    T, dphi, f = dividebyair3(dir, 0)
    j=0
    fi=-1000
    for i in range(2, len(dphi)):
        if abs(dphi[i])>abs(dphi[i-1])-0.1 and  abs(dphi[i])<abs(dphi[i-1])+0.1 and dphi[i]<dphi[i-1]-5.8 and dphi[i-1]>np.pi-0.1 and f[i]>fi+70:
            j+=1
            fi=f[i]
            print(fi)
        dphi[i-2]=dphi[i-2]+2*j*np.pi
        if dphi[i-2]>2*j*np.pi+np.pi/2 and j>0 and f[i]<fi+100:
            dphi[i-2]=dphi[i-2]-2*np.pi
    f=np.array(f)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f[1::]/10**3, T[1::])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.ylabel("Transmittance")
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(f[1:len(f)-2]/10**3, dphi[1:len(f)-2])
        plt.axvline(x=0.557, linestyle="--", color="k")
        plt.axvline(x=0.752, linestyle="--", color="k")
        plt.axvline(x=0.988, linestyle="--", color="k")
        plt.axvline(x=1.097, linestyle="--", color="k")
        plt.axvline(x=1.113, linestyle="--", color="k")
        plt.axvline(x=1.163, linestyle="--", color="k")
        plt.axvline(x=1.208, linestyle="--", color="k")
        plt.axvline(x=1.229, linestyle="--", color="k")
        plt.grid()
        plt.ylabel("Phase shift (rad)")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return T, dphi, f

def lorentz(f, A, f0, df, b, a):
    return (A*df/2)/((f0-f)**2+(df/2)**2)+b+a*f


def lorentzfit(dir, plot):
    T, dphi, f = dividebyair5(dir, 0)
    P0=[-2.6, 1005, 18, 1.5, -8.8*10**-4]
    f=np.array(f)
    P=curve_fit(lorentz, f, T, P0, maxfev=5000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.plot(f[1:len(f)]/10**3, T[1:len(T)], label="data")
        plt.plot(f/10**3, lorentz(f, *P))
        plt.xlabel("Frequency (THz)")
        plt.legend()
        plt.grid()
        plt.show()
    return P


def lorentzfit2(dir, plot):
    T, dphi, f = dividebyair7(dir, 0)
    P0=[-2.6, 1005, 18, 1.5, -8.8*10**-4]
    f=np.array(f)
    P=curve_fit(lorentz, f, T, P0, maxfev=5000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.plot(f[1:len(f)]/10**3, T[1:len(T)], label="data")
        plt.plot(f/10**3, lorentz(f, *P), label="f0="+str(P[1]/10**3)+" df="+str(P[2]/10**3))
        plt.xlabel("Frequency (THz)")
        plt.legend()
        plt.grid()
        plt.show()
    return P


def absorp(dir, plot):
    T, dphi, f = dividebyair5(dir, 0)
    A=-np.log10(np.array(T))
    if plot==1:
        plt.figure()
        plt.plot(np.array(f[3:len(f)-20])/10**3, A[3:len(f)-20])
        plt.grid()
        plt.ylabel("Absorption")
        plt.xlabel("Frequency (THz)")
        plt.show()
    return (A, dphi, f)



def lorentzfitabs(dir, plot):
    T, dphi, f = absorp(dir, 0)
    P0=[20, 1050, 20, 0.96, 6.15*10**-4]
    f=np.array(f)
    P=curve_fit(lorentz, f, T, P0, maxfev=5000)[0]
    print(P)
    if plot==1:
        plt.figure()
        plt.plot(f[1:len(f)-20]/10**3, T[1:len(T)-20], label="data")
        plt.plot(f/10**3, lorentz(f, *P), label="f0="+str(P[1]/10**3)+" df="+str(P[2]/10**3))
        plt.xlabel("Frequency (THz)")
        plt.legend()
        plt.grid()
        plt.show()
    return P

lorentzfitabs(dir5+NiO900, 1)

##Temporal analysis (Inverse FFT)

def fft(dir, plot):
    f, C= openfile(dir, 0)
    yf=np.fft.fft(C)
    xf=np.arange(0, len(f)/(2*f[len(f)-1]), 1/(2*f[len(f)-1]))
    amp=np.abs(yf)
    angle=np.angle(yf)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(xf[0:len(xf)], np.abs(yf))
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(xf[0:len(xf)], np.angle(yf))
        plt.grid()
        plt.show()
    return (xf, amp, angle)


def fft2(dir, plot):
    f, C, f2, C2= open1(dir, 0)
    yf=scipy.fft.ifft(C)
    C2.reverse()
    yf2=scipy.fft.ifft(C2)
    xf=np.arange(0, len(f)/(2*f[len(f)-1]), 1/(2*f[len(f)-1]))
    xf2=np.arange(0, len(f2)/(2*f2[0]), 1/(2*f2[0]))
    amp1=np.abs(yf)
    amp2=np.abs(yf2)
    angle1=np.angle(yf)
    angle2=np.angle(yf2)
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(xf[0:len(xf)], np.abs(yf), label="air")
        plt.plot(xf2[0:len(xf2)],np.abs(yf2), label="lactose")
        plt.legend()
        plt.grid()

        ax=plt.subplot(212)
        plt.plot(xf[0:len(xf)], np.angle(yf))
        plt.plot(xf2[0:len(xf2)],np.angle(yf2))
        plt.grid()
        plt.show()
    return (xf, amp1, angle1, xf2, amp2, angle2)

def ampfft(dir1, plot):
    xf1, amp1, angle1= fft(dir+filename, 0)
    xf2, amp2, angle2= fft(dir1, 0)
    j=10
    A=[]
    A2=[]
    f=[]
    f2=[]
    for i in range(0, int(len(amp1)/j)*j, j):
        A.append(np.max(amp1[i:i+j]))
    for i in range(0, int(len(amp2)/j)*j, j):
        A2.append(np.max(amp2[i:i+j]))
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(A, label="air")
        plt.plot(A2)
        plt.grid()
        plt.legend()


        ax=plt.subplot(212)
        plt.grid()
        T=np.array(A2)/np.array(A)
        plt.plot(T)
        one=np.linspace(1, 1, len(f))
        plt.plot(one, "k--")
        plt.ylabel("Relative spectrum")
        plt.show()
    return(f, A, f2, A2)

def ampfft2(dir1, plot):
    xf1, amp1, angle1= fft(dir2+air2, 0)
    xf2, amp2, angle2= fft(dir1, 0)
    j=10
    A=[]
    A2=[]
    f=[]
    f2=[]
    for i in range(0, int(len(amp1)/j)*j, j):
        A.append(np.max(amp1[i:i+j]))
        f.append(xf1[i+int(j/2)])
    for i in range(0, int(len(amp2)/j)*j, j):
        A2.append(np.max(amp2[i:i+j]))
        f2.append(xf2[i+int(j/2)])
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f, A, label="air")
        plt.plot(f2, A2)
        plt.grid()
        plt.legend()


        ax=plt.subplot(212)
        plt.grid()
        T=np.array(A2)/np.array(A)
        plt.plot(f, T)
        one=np.linspace(1, 1, len(f))
        plt.plot(f, one, "k--")
        plt.ylabel("Relative spectrum")
        plt.show()
    return(f, A, f2, A2)


def ampfft3(dir, plot):
    xf1, amp1, angle1, xf2, amp2, angle2= fft2(dir, 0)
    j=130
    A=[]
    A2=[]
    f=[]
    f2=[]
    for i in range(0, int(len(amp1)/j)*j, j):
        A.append(np.max(amp1[i:i+j]))
        f.append(xf1[i+int(j/2)])
    for i in range(0, int(len(amp2)/j)*j, j):
        A2.append(np.max(amp2[i:i+j]))
        f2.append(xf2[i+int(j/2)])
    if plot==1:
        plt.figure()
        ax=plt.subplot(211)
        plt.plot(f, A, label="air")
        plt.plot(f2, A2, label="lactose")
        plt.grid()
        plt.legend()


        ax=plt.subplot(212)
        plt.grid()
        T=np.array(A2)/np.array(A)
        plt.plot(f, T)
        one=np.linspace(1, 1, len(f))
        plt.plot(f, one, "k--")
        plt.ylabel("Relative spectrum")
        plt.show()
    return(f, A, f2, A2)














