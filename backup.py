"""
Backup of all kinds of random code
"""

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

def find_peaks_cwt_old(bins, width, snr, peak_separation = 0, gap_scale = 0.05, gap_thresh = 2):
    wavelet = Ricker()

    min_scale = np.log2(4*width)
    max_scale = np.log2(bins['edge'][-1] - bins['edge'][0])
    
    arr = bins['height'].astype(np.float)
    arr[arr < 1.0] = 1
    arr = np.log(arr)

    noise_window = int(0.5/width) # TODO: choose a proper noise window

    wa = WaveletAnalysis(arr, wavelet=wavelet, dt=width)
    scales = wa.scales = 2**(np.arange(min_scale, max_scale, gap_scale))

    power = np.real(wa.wavelet_transform)
    #power[power < 1e-8] = 1e-8
    #power = np.log(power)

    def delete_ridge(row, col):
        ridge['loc'][row, col] = -1
        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1    
            
            ridge['loc'][row, col] = -1


    dtype = np.dtype([('power', np.float64),
                      ('peak', np.bool),
                      ('from', np.int64),
                      ('gap', np.int64),
    #                  ('length', np.int),
                      ('max', np.float64),
                      ('loc', np.int64)])

    ridge = np.ndarray(shape=power.shape, dtype=dtype)
    ridge['peak'] = False
    ridge['gap'] = 0
    ridge['max'] = 0
    ridge['from'] = -1
    ridge['loc'] = -1
    ridge['power'] = power

    last_peaks = np.empty((0), dtype=np.int64)
    dtype = np.dtype([('row', np.int64),
                      ('col', np.int64),
                      ('max', np.float64), 
                      ('loc', np.int64)])
    start_ridges = np.empty(bins.shape, dtype=dtype)
    start_ridges_idx = 0
    for index, scale in np.ndenumerate(scales):
        window_size = int(max(1, scale/width))
        print(window_size)
        
        # find the peaks on the current level
        row = index[0]
        peaks = find_peaks_level(power[row,:], window_size)
        ridge['peak'][row,peaks] = True
        
        # match with the previous levels or make new ridge
        #if last_peaks is not None:
        points = np.searchsorted(last_peaks, peaks)
        last_peaks_length = len(last_peaks)
        for ind, ins_idx in np.ndenumerate(points): 
            min_i = -1
            min_v = window_size+1

            if ins_idx != 0 and last_peaks[ins_idx-1] != -1 and peaks[ind]-last_peaks[ins_idx-1] < min_v: 
                min_v = peaks[ind]-last_peaks[ins_idx-1]
                min_i = ins_idx-1
            if ins_idx != last_peaks_length and last_peaks[ins_idx] != -1 and last_peaks[ins_idx]-peaks[ind] < min_v:
                min_v = last_peaks[ins_idx]-peaks[ind]
                min_i = ins_idx

            # TODO: improve use of gap for better matching
            if min_v <= window_size//2:
                ridge['from'][row,peaks[ind]] = last_peaks[min_i]
                ridge['gap'][row,peaks[ind]] = 0
                ridge['max'][row,peaks[ind]] = max(power[row,peaks[ind]], ridge['max'][row-1,last_peaks[min_i]])
                ridge['loc'][row,peaks[ind]] = ridge['loc'][row-1,last_peaks[min_i]]
                
                last_peaks[min_i] = -1 # TODO: check if multi peak matching occurs and how to handle
            else:
                ridge['from'][row,peaks[ind]] = -1
                ridge['gap'][row,peaks[ind]] = 0
                ridge['max'][row,peaks[ind]] = power[row,peaks[ind]]
                ridge['loc'][row,peaks[ind]] = peaks[ind]
        
        # update all non-matched peaks
        for i, idx in np.ndenumerate(last_peaks):
            if idx == -1: continue
            
            if ridge['peak'][row,idx]: 
                print("IMPOSSIBLE")
                continue
                
            ridge['gap'][row,idx] = ridge['gap'][row-1,idx]+1
            ridge['from'][row,idx] = idx
            ridge['max'][row,idx] = ridge['max'][row-1,idx]
            ridge['loc'][row,idx] = ridge['loc'][row-1,idx]
            
            if ridge['gap'][row,idx] > gap_thresh:
                start_ridges[start_ridges_idx]['row'] = row
                start_ridges[start_ridges_idx]['col'] = idx
                start_ridges[start_ridges_idx]['max'] = ridge['max'][row, idx]
                start_ridges[start_ridges_idx]['loc'] = ridge['loc'][row, idx]
                
                start_ridges_idx = start_ridges_idx + 1    
                last_peaks[i] = -1
            
        # remove the peaks that are matched or over the gap threshold and peaks that are not considered anymore
        last_peaks = last_peaks[last_peaks != -1]
        peaks = peaks[peaks != -1]
        
        # merge the new peaks in
        last_peaks = np.concatenate((last_peaks, peaks))
        last_peaks.sort(kind='mergesort')

    # get last ridge lines
    for i, idx in np.ndenumerate(last_peaks):
        start_ridges[start_ridges_idx]['row'] = len(scales)-1
        start_ridges[start_ridges_idx]['col'] = idx
        start_ridges[start_ridges_idx]['max'] = ridge['max'][-1, idx]
        start_ridges[start_ridges_idx]['loc'] = ridge['loc'][-1, idx]
        start_ridges_idx = start_ridges_idx + 1
        
    # get the start of ridge lines
    start_ridges = start_ridges[:start_ridges_idx]
            
    # correct maxima for snr and push them down deleting any that does not work
    num_points = power[0,:].shape[0]
    tot_noise = np.percentile(abs(power[0,:]), 95)
    for ind, (row, col, ridge_max, loc) in np.ndenumerate(start_ridges):    
        # estimate noise at smallest bin level
        level = 0; #np.where(scales > 10*width)[0][0]
        window_start = max(loc - noise_window, 0)
        window_end = min(loc + noise_window, num_points)
        noise = np.percentile(abs(power[level,window_start:window_end]), 95)
                
        if ridge_max / noise < snr: 
            delete_ridge(row, col)
            start_ridges['row'][ind] = -1
            continue
                
        max_loc = -1
        if np.isclose(ridge_max, power[row,col]):
            ridge['max'][max(0,row-3):row+3, max(0, col-100):col+100] = ridge_max
            max_loc = row
            
        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1
            
            if np.isclose(ridge_max, power[row,col]):
                max_loc = row
                ridge['max'][max(0,row-3):row+3, max(0, col-100):col+100] = ridge_max
            else:
                ridge['max'][row,col] = 0    
                
        # TODO VARY THIS OVER THE SPECTRUM INSTEAD        
        if ridge_max / tot_noise < 4 and scales[max_loc] < 10*width:
            delete_ridge(start_ridges[ind]['row'], start_ridges[ind]['col'])
            start_ridges['row'][ind] = -1
            
    start_ridges = start_ridges[start_ridges['row'] != -1]
    start_ridges = np.sort(start_ridges, order="loc")
    
    max_ridges = np.sort(start_ridges, order="max")[::-1]
    for rdg in max_ridges:
        bef, idx, aft = np.searchsorted(start_ridges['loc'], [rdg['loc']-peak_separation/width, rdg['loc'], rdg['loc']+peak_separation/width])
        
        if bef == aft or idx == start_ridges.shape[0] or start_ridges[idx]['loc'] != rdg['loc']: continue
        start_ridges['row'][bef:idx] = -1
        start_ridges['row'][idx+1:aft] = -1
        start_ridges = start_ridges[start_ridges['row'] != -1]
    
    return wa.time, wa.scales, ridge, start_ridges

# FIND CORRELATION LINES
from apt_peaks.algs import find_correlation_lines
mult = data_multi.copy()
ridge, start_ridges = find_correlation_lines(ht, xe[1]-xe[0])

for rdg in start_ridges:
    row = rdg['row']
    col = rdg['col']

    y = np.zeros(rdg['length'])
    x = np.zeros(rdg['length'])
    y[0] = ridge['power'][row, col]
    x[0] = col
    k = 1
    while ridge['from'][row, col] != -1:
        col = ridge['from'][row, col]
        row = row-1
        
        #print(ridge['power'][row, col], rdg['max']/2)
        if ridge['power'][row, col] > rdg['max']/10: 
            #print("OK")
            break

        #ridge['max'][row, min(0, col-1):col+1] = -1    
            
        if x[k-1] == col: continue
        y[k] = ridge['power'][row, col]
        x[k] = col
        k = k+1
    #if k > 99: break

k = 0
corr_x = np.zeros(mult.shape)
corr_y = np.zeros(mult.shape)
for i, el in np.ndenumerate(mult):
    i = i[0]
    if el['Mhit'] == 0: continue
        
    if el['m'] > 100: continue
        
    for j in range(1,el['Mhit']):
        if mult[i+j]['m'] > 100: continue
        x_bin = int(el['m']/(xe[1]-xe[0]))
        y_bin = int(mult[i+j]['m']/(ye[1]-ye[0]))
        
        #print(xe[x_bin], xe[x_bin+1], el['m'])
        #break

        if ridge['max'][y_bin-1, x_bin-1] < 0:
            corr_x[k] = el['m']
            corr_y[k] = mult[i+j]['m']
            k = k+1

            mult[i]['m'] = -1
            #break
       
    if mult[i]['m'] == -1:
        for j in range(1,el['Mhit']):
            #pass
            mult[i+j]['m'] = -1
            
#print(mult[i]['m'])
                        
corr_x = corr_x[:k]
corr_y = corr_y[:k]

print(len(mult))
mult = mult[mult['m'] > -1]
print(len(mult))

plt.scatter(corr_x, corr_y)
X, Y = np.meshgrid(xe[:-1], ye[:-1])
plt.contourf(X, Y, ridge_log, 100, interpolationt="none")
plt.axes().set_aspect('equal', 'datalim')

plt.figure(7)
plt.clf()

mult_bins, width = bin_data(mult['m'], 0.002)
mult_bins = zero_extend(mult_bins, width)
mult_bins = cap_bins(mult_bins, 0, 100)

plt.plot(mult_bins['edge']+width/2,mult_bins['height'])
plt.yscale('log')

for i, el in enumerate(mult):
    if el['Mhit'] == 0: continue
    if el['m'] > 100: continue
    
    first_valid = False
    second_valid = False
        
    ins_idx = np.searchsorted(peak_intervals, el['m'])
    if (ins_idx % 2) == 1:
        #print(peak_intervals[ins_idx-1], peak_intervals[ins_idx])
        first_valid = True
        
    for j in range(1,el['Mhit']):
        ins_idx = np.searchsorted(peak_intervals, mult[i+j]['m'])
        if (ins_idx % 2) == 1:
            second_valid = True
            
        if not first_valid:
            mult[i+j]['m'] = -1
       
    if not second_valid:
        mult[i]['m'] = -1
        
# PEAK MODELING
from scipy import optimize
from scipy import special
#fit_bins = cap_bins(raw_bins, 33, 36)
#fit_bins = cap_bins(raw_bins, 63.2, 64)

def fit_func(x, A, B):
    #x = x.astype(np.complex64)
    return A-1/0.05*np.sqrt(1+B*np.sqrt(x))

def exp_func(x, A, B):
    m = 2
    He_n = special.hermitenorm(m)
    gamma = special.gamma
    ricker_const = (-1) ** (m + 1) / gamma(m + 0.5) ** .5
    ricker_func = He_n(x) * np.exp(-(x ** 2 / 2) * 500)
    
    Un_n = lambda t: 0.5 * (np.sign(t) + 1)
    return Un_n(x) * (np.exp(A-1/0.05*np.sqrt(1+B*np.sqrt(np.abs(x))))) + 1000 * ricker_const * ricker_func

ranges = [(12.0, 14), (19.98,26), (32,37), (47, 51), (63, 68)]

mod_bins = raw_bins.copy().astype(np.dtype([('height', np.float64), ('edge', np.float64)]))
mod_bins['height'] -= exp_func(mod_bins['edge']+width/2-19.98, 28, 0.2)
mod_bins['height'][mod_bins['height'] < 1e-1] = 1e-1

plt.figure(1)
plt.clf()
plt.plot(mod_bins['edge']+width/2, mod_bins['height'], color='y')
#plt.plot(mod_bins['edge']+width/2, exp_func(mod_bins['edge']+width/2-19.98, 28, 0.2), color='c')
plt.yscale('log')

for r in ranges:
    fit_bins = cap_bins(mod_bins.copy(), r[0], r[1])
    
    arr = fit_bins['height'].astype(np.float)
    arr[arr < 1.0] = 1
    arr = np.log(arr)
    
    #B, A = np.polyfit((fit_bins['edge']+width/2)+1), np.log(fit_bins['height']), 1)
    popt, pcov = optimize.curve_fit(fit_func, fit_bins['edge']+width/2-r[0], arr)
    A, B = popt
    plt.plot(fit_bins['edge']+width/2, exp_func(fit_bins['edge']+width/2-r[0], A, B), color='r')
    plt.annotate(B, xy=(r[0]+1, max(fit_bins['height'])))
    print(A, B)
    
# PEAK DECONVOLUTION
from scipy import signal
x = np.arange(-10, 0, 0.1)#raw_bins['edge']+width/2
divisor = np.exp(-20*np.sqrt(1+0.2*np.sqrt(np.abs(x)))) #np.exp(-0.75*raw_bins['edge']+width/2)
divisor = np.concatenate((np.zeros(divisor.shape), divisor))
plt.figure(2)
plt.clf()
plt.plot(divisor)
dec, remainder = signal.deconvolve(raw_bins['height'], divisor[::-1]) #, mode="same"

print(dec.shape)
plt.figure(1)
plt.clf()
plt.plot(raw_bins['edge']+width/2, raw_bins['height'], color='y')
plt.plot( dec, color='r') # raw_bins['edge']+width/2,
plt.yscale('log')

#ridge['peak'] = False

# SHOW MAXIMA
# push down the maxima
for row, col in start_ridges[::-1]:
    ridge_max = ridge['max'][row, col]
    while ridge['from'][row, col] != -1:
        #if not ridge['peak'][row, col] or ridge['max'][row, col] > ridge_max: 
        #    print("ERROR")
                  
        ridge['max'][row, col] = ridge_max
        col = ridge['from'][row, col]
        row = row-1
    
plt.figure(1)
plt.clf()
f, axarr = plt.subplots(2, sharex=True, num=1)
axarr[0].plot(raw_bins['edge']+width/2, arr, color='b')
#axarr[0].set_yscale('log', basey=10)
#axarr[0].contourf(T, S, power, 100)
#axarr[0].set_yscale('log', basey=2)
axarr[1].contour(T, S, ridge['max'], 10, cmap='Greys', interpolation='none')
axarr[1].set_yscale('log', basey=2)

plt.figure(2)
plt.clf()
#print(np.nonzero(ridge['peak']))
plt.imshow(ridge['peak'], cmap='Greys', extent=[0, 100, -10, 10], origin='lower')

# INTEGRATION TESTS
from scipy import integrate

plt.figure(3)
plt.clf()
t = np.arange(-5,10, 0.01)
y = apt_func(t, 0, 0.2)

print(2**-4, width)

g = signal.gaussian(t.shape[0], (2**-5)/width)
norm = np.trapz(g)
g /= norm

y = signal.fftconvolve(y, g, mode="same")
plt.plot(t, y)

def fit_func(x, A, B):
    #x = x.astype(np.complex64)
    return A-1/0.05*np.sqrt(1+B*np.sqrt(x))

def old_apt_func(x, d, C, B = 20):
    y1 = np.zeros(np.count_nonzero(x < d))
    y2 = np.exp(-B*np.sqrt(1+C*np.sqrt(x[x>=d]-d)))

    y = np.concatenate((y1, y2))
    g = signal.gaussian(t.shape[0], (2**-5)/width)
    norm = np.trapz(g)
    g /= norm
    
    return signal.fftconvolve(y, g, mode="same")
    #return 

def gauss_func(m, m_0, S):
    s = S*m_0
    print(1/(s**2),m_0,m)
    c = 1/(np.sqrt(2*np.pi*s**2))
    np.seterr(all="ignore")
    f = np.exp(-1/2*(m/s)**2)
    return c*f

def apt_func(m, m_0, B, C):
    y1 = np.zeros(np.count_nonzero(m < m_0))
    y2 = (B**4*C**2)/(8*(3+B*(3+B)))*np.exp(B-B*np.sqrt(1+C*(np.sqrt(m[m>=m_0])-np.sqrt(m_0))))

    y = np.concatenate((y1, y2))
    return y

def apt_func2(m, m_0, A1, B1):
    t = np.sqrt(m[m>=m_0]-m_0)
    y1 = np.zeros(np.count_nonzero(m < m_0))
    y2 = A1**(np.exp(-B1*t)) #+A1**(B1/(B2-B1)*(np.exp(-B1*t)-np.exp(-B2*t)))*A2**np.exp(-B2*t)
    
    y = np.concatenate((y1, y2))
    return y

def mod_func2(m, m_0, S, A1, B1):
    g = gauss_func(m-m[len(m)/2]+1e-8, m_0, S)
    #a = apt_fit
    #ind = np.nonzero(m >= m_0)[0][0]
    #a = np.pad(a, [ind-200, 0], 'minimum')
    #a = a[:len(m)]
    a = apt_func2(m, m_0, A1, B1)
    return signal.fftconvolve(a, g, mode="same")*(m[1]-m[0])

def mod_func(m, m_0, S, A1, B1):
    g = gauss_func(m-m[len(m)/2], m_0, S)
    #a = apt_fit
    #ind = np.nonzero(m >= m_0)[0][0]
    #a = np.pad(a, [ind-200, 0], 'minimum')
    #a = a[:len(m)]
    a = apt_func(m, m_0, A1, B1)
    return signal.fftconvolve(a, g, mode="same")*(m[1]-m[0])
    
#mult = mult[:100]
from matplotlib import colors
import matplotlib.cm as cm
from apt_peaks.peaks import find_ridge_lines

#start_ridges = np.sort(start_ridges, order=['max'])[::-1]
start_ridges = np.sort(start_ridges, order=['loc'])
print(len(start_ridges))

peak_ignore = np.empty(shape=(start_ridges.shape[0],start_ridges.shape[0]*start_ridges.shape[0]), dtype=np.float32)
peak_ignore[:] = -1
peak_ignore_cnt = np.zeros(shape=start_ridges.shape[0], dtype=np.int32)

plt.figure(8)
plt.clf()

for index, lvl in enumerate(peak_intervals):
    if (index % 2) == 1: continue
   
    if peak_intervals[index+1] > 100: continue
    print(peak_intervals[index], peak_intervals[index+1])

    plt.axvspan(peak_intervals[index], peak_intervals[index+1], color='y')

#plt.yscale('sqrt')
#plt.xscale('sqrt')    

#diffs = set()
#diffsps = 100
#for x in range(0, len(start_ridges)):
#    for y in range(0, len(start_ridges)):
#        srx = mass[start_ridges[x]['loc']]
#       sry = mass[start_ridges[y]['loc']]
        #if sry > srx: srx, sry = sry, srx
#        src = np.sqrt(srx)-np.sqrt(sry)
        
#        diffs.add(int(abs(src*diffsps)))
        
#print(diffs)
#return 

cmass = mass
print(mass, cmass)
#print(cmass[start_ridges['loc']])
#return

#print(start_ridges['max'])

for x in range(0, len(start_ridges)):
    for y in range(0, len(start_ridges)):
        if start_ridges[y]['max'] < 25 or start_ridges[x]['max'] < 25: continue
        #print(start_ridges[x]['loc'])
        
        srx = cmass[start_ridges[x]['loc']]
        sry = cmass[start_ridges[y]['loc']]
        #print(srx, sry)
        #if sry > srx: srx, sry = sry, srx
        src = np.sqrt(srx)-np.sqrt(sry)
        if np.isclose(src, 0): continue
            
        sri = np.nonzero(np.sqrt(xe[:-1])-src < 0)[0]
        if len(sri) == 0: sri = 0
        else: sri = sri[-1]
        #print(srx, sry, src, sri)
        
        #print(src)
        plt.plot(xe[sri:-1], (np.sqrt(xe[sri:-1])-src)**2, color='r', linewidth=2)
        
        for i in range(0, len(start_ridges)):
            loc = start_ridges[i]['loc']
            if np.sqrt(cmass[loc]) < src: continue
            sli = np.searchsorted(cmass, (np.sqrt(cmass[loc])-src)**2)
            if sli == len(cmass): continue
            peak_ignore[i][peak_ignore_cnt[i]] = mass[sli]
            peak_ignore_cnt[i] = peak_ignore_cnt[i] + 1

for i in range(0, len(start_ridges)):
    peak_ignore[i] = np.sort(peak_ignore[i])
    print(cmass[start_ridges[i]['loc']], peak_ignore[i][peak_ignore[i] != -1])
    
data_multi['m'] = data[data['Mhit'] != 1]['m']
    
mult = data_multi.copy()
mult['m'] = -1
#mult = mult[:100]

plt.pcolormesh(xe[:-1], ye[:-1], ht, norm=colors.LogNorm())
plt.ylim([0, 100])
plt.xlim([0, 100])
#return 0
#return 0

corr_x = np.zeros(mult.shape)
corr_y = np.zeros(mult.shape)
cnt = 0

tmp_cnt = 0
for i, el in enumerate(mult):
    if el['Mhit'] == 0: continue
                
    for j in range(0,el['Mhit']):
        if data_multi[i+j]['m'] > 210: continue
            
        ins_idx = np.searchsorted(peak_intervals, data_multi[i+j]['m'])
        if (ins_idx % 2) != 1:
            continue
        pk_idx = np.searchsorted(mass[start_ridges['loc']], data_multi[i+j]['m'])
        if pk_idx == len(start_ridges['loc']) or (pk_idx and data_multi[i+j]['m']-mass[start_ridges['loc'][pk_idx-1]] < mass[start_ridges['loc'][pk_idx]]-data_multi[i+j]['m']):
            pk_idx = pk_idx-1
            
        for k in range(0, el['Mhit']):
            if data_multi[i+k]['m'] > 210: continue
              
            ins_idx = np.searchsorted(peak_ignore[pk_idx], data_multi[i+k]['m'])
            #print(len(peak_ignore[ins_idx/2]))

            min_sz = np.inf
            if ins_idx: min_sz = min(min_sz, data_multi[i+k]['m']-peak_ignore[pk_idx][ins_idx-1])
            if ins_idx!=len(peak_ignore[pk_idx]): min_sz = min(min_sz, peak_ignore[pk_idx][ins_idx]-data_multi[i+k]['m'])
                
            #print(min_sz, data_multi[i+k]['m'], peak_ignore[pk_idx][ins_idx-1])
            #break
            if min_sz < 0.75:
                tmp_cnt = tmp_cnt + 1
                continue
            
            mult[i+k]['m'] = data_multi[i+k]['m']

            # extend if almost full
            if cnt+2 > corr_x.shape[0]:
                corr_x = np.pad(corr_x, (0, corr_x.shape[0]), 'constant', constant_values=0)
                corr_y = np.pad(corr_y, (0, corr_x.shape[0]), 'constant', constant_values=0)

            corr_x[cnt] = mult[i+j]['m']
            corr_y[cnt] = mult[i+k]['m']

            cnt = cnt+1

            #corr_x[cnt] = np.sqrt(mult[i+k]['m'])
            #corr_y[cnt] = np.sqrt(mult[i+j]['m'])

print(tmp_cnt)
#return 0
corr_x = corr_x[:cnt]
corr_y = corr_y[:cnt]

#plt.scatter(corr_x, corr_y)

ht2, xe2, ye2 = np.histogram2d(corr_x, corr_y, np.arange(0,10, 0.01)**2)
ht2 = np.transpose(ht2)

#plt.pcolormesh(xe2[:-1], ye2[:-1], ht2, norm=colors.LogNorm())
#plt.plot([10.75, 100], [0, 46.5], color='m', linestyle='-', linewidth=1)
#plt.plot([42, 69], [69, 42], color='r', linestyle='-', linewidth=1)
#print((np.sqrt(xe[:-1])-src))
#plt.plot(xe[sri:-1], (np.sqrt(xe[sri:-1])-src)**2)
#X = Y = None

            
print(len(mult))
mult = mult[mult['m'] > 0]
print('-> ', len(mult))

mult_bins, width = bin_data(mult['m'], 0.001)
mult_bins = cap_bins(mult_bins, 0, 140)
mult_bins = zero_extend(mult_bins, width)

nmass, nscales, nridge, nstart_ridges = find_peaks_cwt(mult_bins, width, 2, 
                                                       min_length_percentage=0.4, 
                                                       peak_separation=0.0, gap_thresh = 2)

plt.figure(2)
plt.clf()
f, axarr = plt.subplots(2, sharex=True, num=2)
axarr[0].plot(mult_bins['edge']+width/2, mult_bins['height'], color='b')
axarr[0].set_yscale('log')

for rdg in nstart_ridges:
    axarr[0].axvspan(nmass[rdg['loc']]-nscales[rdg['max_row']], nmass[rdg['loc']]+nscales[rdg['max_row']], color='y')

nridge_log = nridge['max']
nridge_log[nridge_log < 1e-4] = np.nan
nridge_log = np.log(nridge_log)
axarr[0].set_ylim([10**-1, 10**3])

axarr[1].contourf(nmass, nscales, nridge_log, 100)
axarr[1].scatter(nmass[nstart_ridges['loc']], nscales[nstart_ridges['max_row']])
axarr[1].set_yscale('log', basey=2)  

#delete already known
mstart_ridges = np.ndarray(shape=start_ridges.shape[0]+nstart_ridges.shape[0], dtype=start_ridges.dtype)
mstart_ridges[:start_ridges.shape[0]] = start_ridges
mstart_ridges_cnt = start_ridges.shape[0]

np.seterr(under='ignore')
for rdg in nstart_ridges:    
    ins_idx = np.searchsorted(mass[start_ridges['loc']], nmass[rdg['loc']])
    min_sz = np.inf
    if ins_idx != len(start_ridges) and mass[start_ridges['loc'][ins_idx]] - nmass[rdg['loc']] < min_sz: 
        min_sz = mass[start_ridges['loc'][ins_idx]] - nmass[rdg['loc']]
    if ins_idx and nmass[rdg['loc']] - mass[start_ridges['loc'][ins_idx-1]] < min_sz:
        min_sz = nmass[rdg['loc']] - mass[start_ridges['loc'][ins_idx-1]]
        ins_idx = ins_idx-1
    
    print(min_sz)
    if min_sz > scales[start_ridges['max_row'][ins_idx]]: 
        axarr[0].axvspan((nmass[rdg['loc']]-nscales[rdg['max_row']]), (nmass[rdg['loc']]+nscales[rdg['max_row']]), color='r')
        mstart_ridges[mstart_ridges_cnt] = rdg
        mstart_ridges_cnt = mstart_ridges_cnt+1
        print(nmass[rdg['loc']])
    else:
        axarr[0].axvspan((nmass[rdg['loc']]-nscales[rdg['max_row']]), (nmass[rdg['loc']]+nscales[rdg['max_row']]), color='y')
        
mstart_ridges = mstart_ridges[:mstart_ridges_cnt]
np.sort(mstart_ridges, order=['loc'])
