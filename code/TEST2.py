import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
import matplotlib 


dataair = np.loadtxt("data/Scanning_200GHZ-1320GHZ_air_.txt")
dataair2 = np.loadtxt("data/Scanning_200GHZ-1320GHZ_air_enveloppe.txt")
datasamphold = np.loadtxt("data/Scanning_200GHZ-1320GHZ_sampleholder_.txt")
datasamphold2 = np.loadtxt("data/Scanning_200GHZ-1320GHZ_sampleholder_enveloppe.txt")
datasaph = np.loadtxt("data/Scanning_200GHZ-1320GHZ_saphire_.txt")
datasaph2 = np.loadtxt("data/Scanning_200GHZ-1320GHZ_saphire_enveloppe.txt")
dataSTO = np.loadtxt("data/Scanning_200GHZ-1320GHZ_STO_.txt")
dataSTO22 = np.loadtxt("data/Scanning_200GHZ-1320GHZ_STO_enveloppe.txt")

def cos(f, A, phi):
    return np.exp(A)*np.cos(2.358*f+phi)

def cosfit(f,C,j,p0,name):
    A=[];freq=[];phi=[]
    plt.figure()
    plt.grid()
    plt.plot(f/10**3, C, ".", label="Data")
    for i in range(0, len(f)-j-1, 3):
        P=sc.optimize.curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
        if np.exp(P[0])>0.01:
            A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[1])
            plt.plot(f[i:i+j]/10**3, cos(f[i:i+j], *P), "r")

    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Photocurrent (nA)")
    plt.title(name)
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_M1/Figures/"+name)
    plt.show()
    return (A, phi, freq)

f=dataair[:,3]
A=dataair[:,2]
j=108;P0=[10, 0]
AA,phi,freq=cosfit(f,A,j,P0,name="Absorbtion air 200-1320GHz")
print(AA)
name2="Enveloppe air (200-1320GHz)"

def hl_envelopes_idx(s, dmin=1, dmax=5, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

if __name__ == "__main__":
    x = f
    y = A
    # Now we get the high and low envelopes
    lmin, lmax = hl_envelopes_idx(y, dmin=5, dmax=9, split=False)
    # Plotting
    #plt.plot(x, y, label='Original Data')
    plt.plot(x[lmax], y[lmax], label='Enveloppe Air 200-1320GHz')
    plt.xscale("log")
    plt.xlabel("Frequencies (in GHz)")
    plt.ylabel("Photocurrent (in nA)")
    plt.title("Enveloppe air (200-1320GHz)")
    plt.legend()
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_M1/Figures/"+name2)
    plt.show()
    plt.close()

TXT=[2,]
TXT[0,:]=x[lmax]
TXT[1,:]=y[lmax]
np.savetxt("C:/Users/MAISON/Desktop/STAGE_M1/Figures/Dataenvair.txt",TXT)



