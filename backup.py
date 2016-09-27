# DWT
#plt.figure(2)
#plt.clf()
new_bins = raw_bins.copy()
wav = pywt.Wavelet("sym4")

arr = raw_bins['height'].astype(np.float)
arr[arr < 1] = 1
arr = np.log(arr)
wdec = pywt.wavedec(arr, wav)
print(len(wdec))
#plt.plot(wdec[0]);
#for el in wdec[:1]:
#    el[:] = 0
for el in wdec[5:]:
    #el[:] = 0
    el[abs(el) < 20] = 0
    #el[abs(el) > 50] = 0
#for el in wdec[3:]:
    #el[abs(el) < 5] = 0
#    el[abs(el) > 50] = 0
new_height = pywt.waverec(wdec, wav)
plt.figure(1)
plt.clf()
#plt.yscale('log')
plt.plot(raw_bins['edge']+width/2, arr, color='g')
plt.plot(new_bins['edge']+width/2, new_height, color='r')
plt.xlim(0, 100)

# CWT
from scipy import signal
widths = np.arange(1, 8, 1)
print(widths)
t = np.linspace(0, 1, 200, endpoint=False)
sig = (np.sin(2*np.pi*t*20)*np.exp(-100*np.pi*(t-0.2)**2) 
    + (np.sin(2*np.pi*t*20)+2*np.sin(2*np.pi*t*80))*np.exp(-50*np.pi*(t-0.5)**2)
    + 2*np.sin(2*np.pi*t*80)*np.exp(-100*np.pi*(t-0.8)**2))
plt.figure(4)
plt.clf()
plt.plot(t, sig)
cwtmatr = signal.cwt(sig, signal.ricker, widths)
print(cwtmatr.shape)
plt.figure(3)
plt.imshow(np.real(cwtmatr), extent=(0, 1, 1, 8), cmap='PRGn', aspect='auto');
plt.show()

from apt_peaks.wavelet_functions import HalfDOG, Poisson, Exponential, APTFunction, APTFunction2
from wavelets import WaveletAnalysis, Ricker, Morlet

#widths = np.arange(1, 8, 0.1)
exponential = APTFunction2(1/0.05, 100)
old_exponential = APTFunction(1/0.05, 100)
poisson = Poisson(2)
poisson1 = Poisson(1)
halfricker = HalfDOG(2)

wavelet = APTFunction2(1/0.05, 100) # Ricker() 

#base = 2.0**-3
#scales = base*2.0**(np.arange(-8, 0, 1))
arr = raw_bins['height'].astype(np.float)
#arr[arr < 1.0] = 1
#arr = np.log(arr)

wa = WaveletAnalysis(arr, wavelet=wavelet, dt=width)
wa.scales = 2**(np.arange(-8, 10, 0.1))
#wa.scales = wa.scales/4

# wavelet power spectrum
power = np.real(wa.wavelet_transform)
#power[power < 0.01] = np.nan
#power = wa.wavelet_power
#power[-50:] = 0
#wa.wavelet_power = power
power[power < 1e-8] = np.nan
power = np.log(power)

# scales 
scales = wa.scales

# associated time vector
t = wa.time

# reconstruction of the original data
#rx = wa.reconstruction()
#rx[rx < 1] = 1

T, S = np.meshgrid(t, wa.scales)

plt.figure(1)
plt.clf()
f, axarr = plt.subplots(2, sharex=True, num=1)
axarr[0].plot(raw_bins['edge']+width/2, arr, color='y')
#axarr[0].plot(raw_bins['edge']+width/2, 3*(10**2)*exponential(raw_bins['edge']+width/2-19.97, 20), color='r')
#axarr[0].plot(raw_bins['edge']+width/2, 3*(10**2)*exponential(raw_bins['edge']+width/2-19.97, 40), color='b')
#axarr[0].plot(raw_bins['edge']+width/2, 3*(10**2)*old_exponential(raw_bins['edge']+width/2-19.97, 15), color='b')
#axarr[0].plot(raw_bins['edge']+width/2, 3*(10**2)*exponential(raw_bins['edge']+width/2-19.97, 15), color='g')
#axarr[0].plot(raw_bins['edge']+width/2, 600000*poisson(raw_bins['edge']+width/2-20, 0.8), color='k')
#axarr[0].plot(raw_bins['edge']+width/2, 200000*poisson1(raw_bins['edge']+width/2-20, 0.1), color='b')
#axarr[0].plot(raw_bins['edge']+width/2, 10000*halfricker(raw_bins['edge']+width/2-20, 1.5), color='g')

axarr[0].set_yscale('log', basey=10)
#axarr[1].contourf(T, S, power, 100)
axarr[1].set_yscale('log', basey=2)

#plt.plot(raw_bins['edge']+width/2, raw_bins['height'], color='b')
plt.contourf(T, S, power, 100)
plt.yscale('log', basey=2)


