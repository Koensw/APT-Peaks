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
