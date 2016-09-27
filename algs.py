import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import sortedcontainers

def estimate_baseline(bins, width, window, max_baseline):
    bst = sortedcontainers.SortedSet()
    
    window_size = window/(bins['edge'][1] - bins['edge'][0])/2
    window_size = int(max(1, window_size))
        
    base_points = []
        
    cnt = 0
    for i, el in np.ndenumerate(bins):
        iel = el.tolist()
        bst.add(iel)
        
        if i[0] >= window_size: 
            rel = bins[i[0]-window_size]
            bst.remove(rel.tolist())
        
        if i[0] >= window_size and (i[0] % window_size) == 0:
            if bst[len(bst)//2][0] < max_baseline: 
                base_points.append(bst[len(bst)//4])
                
    base_points = np.array(base_points, dtype = bins.dtype)
    baseline = np.interp(bins['edge']+width/2, base_points['edge']+width/2, base_points['height'])
    return baseline
                

def find_peaks(bins, peak_diff = 0.1, peak_amount = None, min_peak_size = None):    
    bst = sortedcontainers.SortedSet()
    
    window_size = peak_diff/(bins['edge'][1] - bins['edge'][0])/2
    window_size = int(max(1, window_size))
    
    mask = np.zeros(shape=bins.shape, dtype=(np.bool))
    
    cnt = 0
    cnt_el = None
    tot_height = 0
    for i, el in np.ndenumerate(bins):
        iel = el.tolist()
        tot_height += el['height']
        bst.add(iel)
        
        if i[0] >= window_size: 
            rel = bins[i[0]-window_size]
            tot_height -= rel['height']
            bst.remove(rel.tolist())
        
        if bst[-1] == iel:
            cnt = 0
            cnt_el = iel
            
        if not cnt_el is None: 
            if bst[-1] != cnt_el:
                cnt = 0
                cnt_el = None
            else: cnt += 1
            
            # peaks falls within window size and passes over minimum size
            if cnt == window_size and cnt_el[0] > min_peak_size:
                mask[i[0]-cnt+1] = True
        
    return np.sort(bins[mask])[:peak_amount:-1]
    
def find_peaks_level(bins, window_size, precision = 10**6, initialization_width = 10):    
    bst = sortedcontainers.SortedSet()
    
    mask = np.zeros(shape=bins.shape, dtype=(np.bool))
    pos_peaks = signal.argrelmax(bins, order=min(initialization_width, max(1, window_size//2)))[0]
       
    cnt = 0
    cnt_el = None
    bef = 0
    for i, ind in np.ndenumerate(pos_peaks):
        ht = bins[ind]
        if np.isnan(ht): ht = 0
        ht = int(ht * precision)
        
        bst.add((ht, ind))
        
        while ind - pos_peaks[bef] > window_size:            
            inp = pos_peaks[bef]
            htp = bins[inp]
            if np.isnan(htp): htp = 0
            htp = int(htp * precision)
            
            if cnt_el == (htp, inp): mask[inp] = True
            
            bst.remove((htp, inp))
            
            bef = bef+1
        
        if bst[-1][0] == ht:
            cnt = 0
            cnt_el = bst[-1]
            
        if cnt_el and not bst[-1] == cnt_el: cnt_el = None
            
    if cnt_el and cnt_el[1] == pos_peaks[-1]: mask[pos_peaks[-1]] = True
                 
    return np.nonzero(mask)[0]
        
from apt_peaks.algs import find_peaks_level
from scipy import signal
from apt_peaks.wavelet_functions import HalfDOG, Poisson, APTFunction2
from wavelets import WaveletAnalysis, Ricker, Morlet

def find_peaks_cwt(bins, width, snr, peak_separation = 0, gap_scale = 0.05, gap_thresh = 2):
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