import numpy as np
import matplotlib.pyplot as plt

# This code plotes data from files in the directory

# Read data from files
data_calib = np.loadtxt('Scanning_100GHZ-1320GHZ_air.txt')
data_calib_enveloppe = np.loadtxt('Scanning_100GHZ-1320GHZ_air_enveloppe.txt')
data_air_enveloppe = np.loadtxt('Scanning_900GHZ-1200GHZ_air_enveloppe.txt')
data_Nio_enveloppe = np.loadtxt('Scanning_900GHZ-1200GHZ_NIO_enveloppe.txt')
data_air = np.loadtxt('Scanning_900GHZ-1200GHZ_air.txt')
data_Nio = np.loadtxt('Scanning_900GHZ-1200GHZ_NIO.txt')

plt.figure()
plt.plot(data_calib_enveloppe[:,0], data_calib_enveloppe[:,1], '-')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.savefig('THz_Photocurrent_vs_Frequency_Calib_enveloppe.pdf')
plt.close()


plt.figure()
plt.plot(data_air_enveloppe[:,0], data_air_enveloppe[:,1], '-')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.savefig('THz_Photocurrent_vs_Frequency_Air_enveloppe.pdf')
plt.close()

plt.figure()
plt.plot(data_Nio_enveloppe[:,0], data_Nio_enveloppe[:,1], '-')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency NiO')
plt.savefig('THz_Photocurrent_vs_Frequency_NiO_enveloppe.pdf')
plt.close()

# We pot the forth column against the third column
# We will use log scale in the y axis

plt.figure()
plt.plot(data_calib[:,3], data_calib[:,2], '-', label='Current')
plt.plot(data_calib_enveloppe[:,0], data_calib_enveloppe[:,1], 'r-',label = 'Enveloppe')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.legend()
plt.savefig('THz_Photocurrent_vs_Frequency_Calib.pdf')
plt.close()

plt.figure()
plt.plot(data_air[:,3], data_air[:,2], '-', label='Current')
plt.plot(data_Nio_enveloppe[:,0], data_Nio_enveloppe[:,1], 'r-',label = 'Enveloppe')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.legend()
plt.savefig('THz_Photocurrent_vs_Frequency_Air.pdf')
plt.close()

plt.figure()
plt.plot(data_Nio[:,3], data_Nio[:,2], '-', label='Current')
plt.plot(data_air_enveloppe[:,0], data_air_enveloppe[:,1], 'r-',label = 'Enveloppe')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency NiO')
plt.legend()
plt.savefig('THz_Photocurrent_vs_Frequency_NiO.pdf')
plt.close()

# Now enveloppes of NiO and Air to compare them 
plt.figure()
plt.plot(data_air_enveloppe[:,0], data_air_enveloppe[:,1], '-', label='Air')
plt.plot(data_Nio_enveloppe[:,0], data_Nio_enveloppe[:,1], 'r-',label = 'NiO')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.legend()
plt.savefig('THz_Photocurrent_vs_Frequency_Air_NiO_enveloppe.pdf')
plt.close()


# Pour
plt.figure()
plt.plot(data_air_enveloppe[:,0], data_air_enveloppe[:,1]-data_Nio_enveloppe[:,1], '-')
plt.xlabel('Frequency (GHz)')
plt.ylabel('THz Photocurrent (nA)')
plt.title('THz Photocurrent vs Frequency in Air')
plt.legend()
plt.savefig('THz_Photocurrent_vs_Frequency_Air_NiO_resta.pdf')
plt.close()



