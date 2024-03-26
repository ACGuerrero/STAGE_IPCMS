import numpy as np 
import scipy as sc 
import matplotlib.pyplot as plt 

def cos(f, A, phi):
    return np.exp(A)*np.cos(2.358*f+phi)

def cos2(f,phi):
    return np.cos(f+phi)

def cosfit(f,C,j,p0,name):
    A=[];freq=[];phi=[]
    for i in range(0, len(f)-j-1, 3):
        P=sc.optimize.curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
        if np.exp(P[0])>0.01:
            A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[1])
    return (A, phi, freq)

def cosfit2(f,C,j,p0,name):
    freq=[];phi=[]
    for i in range(0, len(f)-j-1, 3):
        P=sc.optimize.curve_fit(cos, f[i:i+j], C[i:i+j], P0, maxfev=5000)[0]
        if np.exp(P[0])>0.01:
            #A.append(np.exp(P[0]))
            freq.append(f[i+int(j/2)])
            phi.append(P[0])
            #w.append(P[2])
            #plt.plot(f[i:i+j]/10**3, cos(f[i:i+j], *P), "r")
        #plt.legend()
        #plt.xlabel("Frequency (THz)")
        #plt.ylabel("Photocurrent (nA)")
        #plt.title(name)
        #plt.savefig("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name)
        #plt.show()
    return (phi, freq)#w)

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

    #NAMES
    name1="Absorbtion glass 200-1320GHz"
    name3="Absobtion glass 200-1320GHz"
    name2="Transmittance of glass (200-1320GHz)"
    name4="Enveloppe of Absorbtion of glass and Si"


    # LOADING DATA
    dataair = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_air.txt")
    dataglass = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_glass.txt")
    dataSi = np.loadtxt("data/Si substrate/Scanning_200GHZ-1320GHZ_Si.txt")

    # EXTRACTING DATA
    f=dataSi[:,3]
    A=dataSi[:,2]
    Amodif = A[A != 0]
    f2=dataair[:,3]
    A2=dataair[:,2]
    A2modif=A2[A2 !=0]
    A2modif=A2modif[:-1]
    #b=dataSi2[:,0]
    #c=dataSi2[:,1]

    # FUNCTION PARAMETERS
    j=108
    P0=[10, 0]

    # APPLY FUNCTION
    AA,phi,freq=cosfit(f,A,j,P0,name1)
    AA3,phi3,freq3=cosfit(f2,A2,j,P0,name3)

    # Now we get the high and low envelopes
    lmin, lmax = hl_envelopes_idx(A, dmin=5, dmax=9, split=False)
    lmin2, lmax2 = hl_envelopes_idx(A2, dmin=5, dmax=9, split=False)

    # CALCULS

    AA=np.array(AA)
    AA3=np.array(AA3)
    AA3interp=np.interp(freq,freq3,AA3)
    A4=AA/AA3interp


    # PLOTTING

    #plt.figure()
    #plt.plot(freq,AA)
    #plt.plot(freq,AA3)
    #plt.show()



    #plt.plot(x, y, label='Original Data')
    #plt.plot(x[lmax], y[lmax], label=name2)
    #AA2,phi,freq=cosfit(x,Y,j,P0,name1)
    #plt.plot(b,c)
    #plt.plot(freq,AA)
    #plt.plot(freq3,AA3)

    #plt.yscale("log")
    # RATIO BETWEEN AMPLITUDE OF Si AND AIR 
    #plt.plot(x,abs(y2/y))
    #plt.yscale("log")
    plt.xlabel("Frequencies (in GHz)")
    plt.ylabel("Transmittance")
    plt.title(name2)
    plt.legend()
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name2)
    #plt.show()
    plt.close()

    plt.figure()
    plt.plot(freq,AA)
    plt.plot(freq,AA3interp)
    plt.xlabel("Frequencies (in GHz)")
    plt.ylabel("Photocurrent (in nA)")
    plt.title(name4)
    plt.yscale("log")
    plt.savefig("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name4)
    #plt.show()
    plt.close()



    tx=np.size(freq)
    ty=np.size(AA)
    TXT=np.zeros((tx,2))
    TXT[:,0]=freq
    TXT[:,1]=AA
    np.savetxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name3,TXT)


    tx2=np.size(freq)
    ty2=np.size(AA3interp)
    TXT2=np.zeros((tx,2))
    TXT2[:,0]=freq
    TXT2[:,1]=AA3interp
    np.savetxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name1,TXT)

    
    tx3=np.size(freq)
    ty3=np.size(A4)
    TXT3=np.zeros((tx,2))
    TXT3[:,0]=freq
    TXT3[:,1]=4
    np.savetxt("C:/Users/MAISON/Desktop/STAGE_IPCMS/code/Figures"+name2,TXT)
    P1=[0]
    

    def Atrans(f, R1, R2, n):
        l = 285e-6 # Keeping l as a floating-point number
        c = 3e8    # Keeping c as a floating-point number
        return (1-R1)*(1-R2)/((1-np.sqrt(R1*R2))**2+4*np.sqrt(R1*R2)*np.sin(2*np.pi*l*n/c*f)**2)   
    
    freq_int = [int(np.round(x)) for x in freq]
    P0=[0.3, 0.3, 3.9]
    freq_int=np.array(freq)*10**9
    P=sc.optimize.curve_fit(Atrans,freq_int,A4,P0,maxfev=100000)[0]
    j=108
    R1=[]
    R2=[]
    n=[]
    freQ=[]

    #for i in range(0, np.size(freq_int)-j-1, 3):
    #    P=sc.optimize.curve_fit(Atrans, freq_int[i:i+j], A4[i:i+j], P0, maxfev=5000)[0]
    #    if np.exp(P[0])>0.01:
    #        R1.append(np.exp(P[0]))
    #        freQ.append(freq_int[i+int(j/2)])
    #        R2.append(P[1])
    #        n.append(P[2])

    print(P)
    plt.figure()
    plt.plot(freq_int/10**12,A4)
    plt.plot(freq_int/10**12,Atrans(freq_int,P[0],P[1],P[2]))
    plt.show()



