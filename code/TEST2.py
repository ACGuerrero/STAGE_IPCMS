import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 
import matplotlib 

datasaphmircea = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/saphire substrate Mircea/Scanning_200GHZ-1320GHZ_saphire.txt")
datasaphmircea2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/saphire substrate Mircea/Scanning_200GHZ-1320GHZ_saphire_enveloppe.txt")
datasampleholder = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/saphire substrate Mircea/Scanning_200GHZ-1320GHZ_saphire.txt")
datasampleholder2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/saphire substrate Mircea/Scanning_200GHZ-1320GHZ_sample_holder_enveloppe.txt")
dataSi = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")
dataSi2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/data/Si substrate/Scanning_200GHZ-1320GHZ_Si_enveloppe.txt")


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
            #w.append(P[2])
            plt.plot(f[i:i+j]/10**3, cos(f[i:i+j], *P), "r")

    plt.legend()
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Photocurrent (nA)")
    plt.title(name)
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name)
    plt.show()
    return (A, phi, freq)#w)

f=dataSi[:,3]
A=dataSi[:,2]
b=dataSi2[:,0]
c=dataSi2[:,1]
j=108;P0=[10, 0]
w=[]
name1="Absorbtion Si 200-1320GHz"
name2="Enveloppe Si (200-1320GHz)"
AA,phi,freq=cosfit(f,A,j,P0,name1)


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
    plt.plot(x[lmax], y[lmax], label=name2)
    plt.plot(b,c)
    plt.plot(freq,AA)
    plt.yscale("log")
    plt.xlabel("Frequencies (in GHz)")
    plt.ylabel("Photocurrent (in nA)")
    plt.title(name2)
    plt.legend()
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name2)
    plt.show()
    plt.close()

tx=np.size(x[lmax])
ty=np.size(y[lmax])
TXT=np.zeros((tx,2))
TXT[:,0]=x[lmax]
TXT[:,1]=y[lmax]
np.savetxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name2,TXT)




