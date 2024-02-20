
import numpy as np 
import matplotlib.pyplot as plt

data = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_100GHZ-1320GHZ_air.txt")
data2 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_100GHZ-1320GHZ_air_enveloppe.txt")
data3 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_900GHZ-1200GHZ_air.txt")
data4 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_900GHZ-1200GHZ_air_enveloppe.txt")
data5 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_900GHZ-1200GHZ_NIO.txt")
data6 = np.loadtxt("C:/Users/MAISON/Desktop/STAGE_M1/Scanning_900GHZ-1200GHZ_NIO_enveloppe.txt")


plt.figure()
plt.plot(np.log(data[:,3]),data[:,2])
plt.title("Calibration 100GHz-1320GHz")
plt.xlabel("Frequencies (GHz)")
plt.ylabel("Photocurrent (nA)")
plt.savefig("Calib100-1320GHz")
plt.close()

plt.figure()
plt.plot(np.log(data2[:,0]),data2[:,1])
plt.savefig("Enveloppe_calib100-1320")
plt.close()

plt.figure()
plt.plot(np.log(data3[:,3]),data3[:,2],label="Absorbtion of Air")
plt.plot(np.log(data4[:,0]),data4[:,1],label="Envelope")
plt.title("Calibration 900GHz-1200GHz")
plt.legend()
plt.xlabel("Frequencies (GHz)")
plt.ylabel("Photocurrent (nA)")
plt.savefig("Calib900-1200")
plt.close()

plt.figure()
plt.plot(data4[:,0],np.log(data4[:,1]))
plt.savefig("Enveloppe_calib900-1200")
plt.close()

plt.figure()
plt.plot(np.log(data5[:,3]),data5[:,2],label="Absorbtion of NiO")
plt.plot(np.log(data6[:,0]),data6[:,1],label="Envelope")
plt.legend()
plt.title("Scanning NiO 900GHz-1200GHz")
plt.xlabel("Frequencies (GHz)")
plt.ylabel("Photocurrent (nA)")
plt.savefig("Ni0 900-1200GHz")
plt.close()

plt.figure()
plt.plot(data6[:,0],np.log(data6[:,1]))
plt.savefig("Envelope_NiO_900-1200")
plt.xlabel("Frequencies (GHz)")
plt.ylabel("Photocurrent (nA)")
plt.close()


x=np.linspace(900,1200,num=225)

data4new = np.interp(x,data4[:,0],data4[:,1])
data6new =np.interp(x,data6[:,0],data6[:,1])
compare=data4new/data6new

plt.figure()
plt.plot(x,data4new)
plt.plot(x,data6new)
plt.figure()
#plt.plot(x,compare)
#plt.yscale('log')
#plt.show()


max1 = np.max(compare[1:110])
freqres=np.argmax(compare[1:110])
print("La fréquence de résonnance correspond à",data4[freqres,0],"GHz")

pas=90
y=int(np.size(data5[:,2])/pas)
max=np.zeros(y)
data55=data5[:,2]
freqmax=np.zeros(y,dtype="int")

for i in range(1,int(y)):
    max[i]=np.max(data55[(i-1)*pas:i*pas])
    freqmax[i]=np.argmax(data55[(i-1)*pas:i*pas])+i*pas

plt.figure()
plt.plot(data5[:,3][freqmax],max,color="red")
plt.plot(data5[:,3],data5[:,2])
plt.savefig("Envelope_NiO_900-1200_Manual")
plt.xlabel("Frequencies (in GHz)")
plt.ylabel("Photocurrent (in nA)")
plt.legend()
plt.show()
