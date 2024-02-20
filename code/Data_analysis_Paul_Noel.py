import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
import matplotlib 

dataair = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_air_.txt")
dataair2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_air_enveloppe.txt")
datasamphold = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_sampleholder_.txt")
datasamphold2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_sampleholder_enveloppe.txt")
datasaph = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_saphire_.txt")
datasaph2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_saphire_enveloppe.txt")
dataSTO = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_STO_.txt")
dataSTO22 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Absorbtion_substrat/Scanning_200GHZ-1320GHZ_STO_enveloppe.txt")

def cos(freq, Amp, phi):
    return np.exp(-Amp)*np.cos((3*np.pi)/4*freq+phi)

def cosfit(f,c,p0,j,name):
    A=[]
    p=[]
    for i in range(0,len(f)-j-1,3):
        popt,pcov = sc.optimize.curve_fit(cos,dataair[i:i+j,3],dataair[i:i+j,2],p0)
        A=popt[0]
        p=popt[1]
        freq=dataair[i+int(j/2),3]
    plt.figure()
    plt.scatter(dataair[:,3],cos(freq,A,p),s=0.5)
    plt.close()
    plt.show()

    return popt, pcov


def cosfit1(f,C,p0,j):
    A=[]
    freq=[]
    phi=[]
    plt.figure()
    plt.grid()
    plt.plot(f/10**3, C, ".", label="Data")
    for i in range(0, len(f)-j-1, 3):
        popt,pcov=sc.optimize.curve_fit(cos, f[i:i+j], C[i:i+j],p0, maxfev=5000)
        if np.exp(popt[0])>0.01:
            A.append(np.exp(popt[0]))
            freq.append(f[i+int(j/2)])
            phi.append(popt[1])
            plt.plot(f[i:i+j]/10**3, cos(f[i:i+j],*popt), "r")
    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Photocurrent (nA)")
    plt.show()
    return (A, phi, freq)

def cosfit2(f,c,p0,j,name):
    popt,pcov = sc.optimize.curve_fit(cos,dataair[:,3],dataair[:,2],p0)
    plt.figure()
    plt.scatter(dataair[:,3],cos(dataair[:,3],popt[0],popt[1]),s=0.5)
    plt.show()
    return popt, pcov

p0=[10,0]#initial guess 
j=128 #width of the local cos 
A,phi,freq = cosfit1(dataair[:,3],dataair[:,2],p0,j)



#A,phi,freq=cosfit(dataair[:,3],dataair[:,2],1)

#plt.figure()
#plt.plot(dataair[:,3],dataair[:,2])
#plt.title("Calibration 200GHz-1320GHz")
#plt.xlabel("Frequencies (GHz)")
#plt.ylabel("Photocurrent (nA)")
#plt.savefig("Calib200-1320GHz")
#plt.show()
#plt.close()

