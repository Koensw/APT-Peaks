import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy import signal, optimize
from numpy.polynomial import polynomial as nppoly
from wavelets import WaveletAnalysis, Ricker, Morlet

import sortedcontainers

from apt_peaks.wavelet_functions import UnbiasedRicker, HalfDOG, Poisson, APTFunction2

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
    
def find_peaks_level(bins, window_size, precision = 10**6, initialization_width = 10, min_peak_height = -np.inf):    
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
            
            if cnt_el == (htp, inp) and cnt_el[0] > min_peak_height*precision: mask[inp] = True
            
            bst.remove((htp, inp))
            
            bef = bef+1
        
        if bst[-1][0] == ht:
            cnt = 0
            cnt_el = bst[-1]
            
        if cnt_el and not bst[-1] == cnt_el: cnt_el = None
            
    if cnt_el and cnt_el[1] == pos_peaks[-1] and cnt_el[0] > min_peak_height*precision: mask[pos_peaks[-1]] = True
                 
    return np.nonzero(mask)[0]
        
def delete_ridge(ridge, row, col):
    ridge['loc'][row, col] = -1
    ridge['max'][row, col] = 0
    ridge['length'][row, col] = 0
    
    while ridge['from'][row, col] != -1:
        col = ridge['from'][row, col]
        row = row-1    
        
        ridge['loc'][row, col] = -1
        ridge['max'][row, col] = 0
        ridge['length'][row, col] = 0
        
def find_ridge_lines(power, width, left_window, right_window, gap_thresh = 2, noise_window = 0.5, min_noise = 0.1):
    left_window_list = np.array(left_window/width, dtype=np.float64, ndmin=1, copy=False)
    right_window_list = np.array(right_window/width, dtype=np.float64, ndmin=1, copy=False)
    
    left_window_list = np.lib.pad(left_window_list, (0, power.shape[0]-left_window_list.shape[0]), 'edge')
    right_window_list = np.lib.pad(right_window_list, (0, power.shape[0]-right_window_list.shape[0]), 'edge') 
    
    noise_window = int(noise_window/width)
 
    dtype = np.dtype([('power', np.float64),
                      ('peak', np.bool),
                      ('from', np.int64),
                      ('gap', np.int64),
                      ('max', np.float64),
                      ('loc', np.int64),
                      ('length', np.int)])

    ridge = np.ndarray(shape=power.shape, dtype=dtype)
    ridge['length'] = 0
    ridge['peak'] = False
    ridge['gap'] = 0
    ridge['max'] = 0
    ridge['from'] = -1
    ridge['loc'] = -1
    ridge['power'] = power

    last_peaks = np.empty((0), dtype=np.int32)
    dtype = np.dtype([('row', np.int32),
                      ('col', np.int32),
                      ('max', np.float32), 
                      ('max_row', np.int32), 
                      ('loc', np.int32),
                      ('length', np.int32),
                      ('noise', np.float32)])
    start_ridges = np.empty(power.shape[1], dtype=dtype)
    start_ridges_idx = 0
    
    #tmp_cnt = 0
    for index, (left_window, right_window) in enumerate(np.c_[left_window_list, right_window_list]):
        #print(left_window, right_window)
        # find the peaks on the current level
        row = index
        peaks = find_peaks_level(power[row,:], int(max(1, left_window+right_window)), min_peak_height = 0)
        #print(len(peaks), row, left_window+right_window)
        ridge['peak'][row,peaks] = True
        
        # match with the previous levels or make new ridge
        points = np.searchsorted(last_peaks, peaks)
        last_peaks_length = len(last_peaks)
        keep_mask = np.ones(last_peaks.shape, dtype=np.bool)
        for ind, ins_idx in np.ndenumerate(points): 
            min_i = -1
            min_v = max(left_window, right_window)+1 
            
            if ins_idx != 0 and peaks[ind]-left_window <= last_peaks[ins_idx-1] <= peaks[ind]+right_window and peaks[ind]-last_peaks[ins_idx-1] < min_v: 
                min_v = peaks[ind]-last_peaks[ins_idx-1]
                min_i = ins_idx-1
            if ins_idx != last_peaks_length and peaks[ind]-left_window <= last_peaks[ins_idx] <= peaks[ind]+right_window and last_peaks[ins_idx]-peaks[ind] < min_v:
                min_v = last_peaks[ins_idx]-peaks[ind]
                min_i = ins_idx
                
            # if min_i != -1: print(peaks[ind]-left_window, last_peaks[min_i], peaks[ind]+right_window)    
                
            # TODO: improve use of gap for better matching
            if min_v <= max(left_window, right_window):
                ridge['from'][row,peaks[ind]] = last_peaks[min_i]
                ridge['gap'][row,peaks[ind]] = 0
                ridge['max'][row,peaks[ind]] = max(power[row,peaks[ind]], ridge['max'][row-1,last_peaks[min_i]])
                ridge['length'][row,peaks[ind]] = ridge['length'][row-1,last_peaks[min_i]]+1
                ridge['loc'][row,peaks[ind]] = ridge['loc'][row-1,last_peaks[min_i]]
                                
                # allow multi peak matching 
                # TODO: check if multi peak matching occurs and how to handle
                keep_mask[min_i] = 0
            else:
                ridge['from'][row,peaks[ind]] = -1
                ridge['gap'][row,peaks[ind]] = 0
                ridge['length'][row,peaks[ind]] = 1
                ridge['max'][row,peaks[ind]] = power[row,peaks[ind]]
                ridge['loc'][row,peaks[ind]] = peaks[ind]
        
        # update all non-matched peaks
        for i, idx in np.ndenumerate(last_peaks):
            if keep_mask[i] == 0: continue
            
            #if ridge['peak'][row,idx]: 
            #    print("IMPOSSIBLE")
            #    continue
                
            ridge['gap'][row,idx] = ridge['gap'][row-1,idx]+1
            ridge['from'][row,idx] = idx
            ridge['max'][row,idx] = ridge['max'][row-1,idx]
            ridge['loc'][row,idx] = ridge['loc'][row-1,idx]
            ridge['length'][row,idx] = ridge['length'][row-1,idx]
            
            if ridge['gap'][row,idx] > gap_thresh:
                start_ridges[start_ridges_idx]['row'] = row
                start_ridges[start_ridges_idx]['col'] = idx
                start_ridges[start_ridges_idx]['max'] = ridge['max'][row, idx]
                start_ridges[start_ridges_idx]['loc'] = ridge['loc'][row, idx]
                start_ridges[start_ridges_idx]['length'] = ridge['length'][row, idx]
                
                start_ridges_idx = start_ridges_idx + 1   
                if start_ridges_idx >= start_ridges.shape[0]: 
                    start_ridges = np.pad(start_ridges, (0, start_ridges.shape[0]), 'edge')
                keep_mask[i] = 0
            
        # remove the peaks that are matched or over the gap threshold
        last_peaks = last_peaks[keep_mask]
        
        # merge the new peaks in
        last_peaks = np.concatenate((last_peaks, peaks))
        last_peaks.sort(kind='mergesort')

    # get last ridge lines
    for i, idx in np.ndenumerate(last_peaks):
        start_ridges[start_ridges_idx]['row'] = power.shape[0]-1
        start_ridges[start_ridges_idx]['col'] = idx
        start_ridges[start_ridges_idx]['max'] = ridge['max'][-1, idx]
        start_ridges[start_ridges_idx]['loc'] = ridge['loc'][-1, idx]
        start_ridges[start_ridges_idx]['length'] = ridge['length'][-1, idx]
        start_ridges_idx = start_ridges_idx + 1
        if start_ridges_idx >= start_ridges.shape[0]: 
            start_ridges = np.pad(start_ridges, (0, start_ridges.shape[0]), 'edge')
        
    # get the start of ridge lines
    start_ridges = start_ridges[:start_ridges_idx]
    
    # find maximum positions and noise
    noise_level = 0;
    num_points = power[0,:].shape[0]
    for ind, (row, col, ridge_max, _, loc, _, _) in np.ndenumerate(start_ridges):            
        #set maximum row 
        if np.isclose(ridge_max, power[row,col]):
            start_ridges[ind]['max_row'] = row
            start_ridges[ind]['loc'] = col

        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1
            
            if np.isclose(ridge_max, power[row,col]):
                start_ridges[ind]['max_row'] = row
                start_ridges[ind]['loc'] = col
        
        # set location
        start_ridges[ind]['loc'] = col
        
        #estimate noise at lowest bin level
        window_start = max(loc - noise_window, 0)
        window_end = min(loc + noise_window, num_points)
        start_ridges[ind]['noise'] = max(min_noise, np.percentile(abs(power[noise_level,window_start:window_end]), 95))
            
    start_ridges = np.sort(start_ridges, order="loc")
    return ridge, start_ridges

def find_peaks_cwt(bins, width, snr, peak_separation = 0, min_length_percentage = 0.5, 
                   noise_window = 0.5, peak_range = (0.5, np.inf), gap_scale = 0.05, gap_thresh = 2):
    # select the wavelet to use - the ricker (mexican hat) wavelet gives good results
    wavelet = Ricker()

    # select the minimum and maximum applicable scale levels
    min_scale = np.log2(4*width)
    max_scale = np.log2(2)
    
    # compute the logarithm of the height as the CWT performs better here on typical spectra
    height_log = bins['height'].astype(np.float)
    height_log += 1 # add one to circumvent the problem of zero height 
    height_log = np.log(height_log)

    # initialize the wavelet library
    wa = WaveletAnalysis(height_log, wavelet=wavelet, dt=width)
    wa.time = bins['edge']+width/2 # use mass to charge as the time scale
    scales = wa.scales = 2**(np.arange(min_scale, max_scale, gap_scale)) # build a set of scales
    # apply the continous wavelet transform and keep the real part (in theory there should be no imaginary part)
    power = np.real(wa.wavelet_transform) 
            
    ridge, start_ridges = find_ridge_lines(power, width, scales/2, scales/2, gap_thresh, noise_window)        
    
    # correct maxima for snr and push them down deleting any that does not work
    #tot_noise = np.percentile(abs(power[0,:]), 95)
    #noise = np.zeros(shape=bins.shape[0], dtype=np.float32)
    for ind, (row, col, ridge_max, max_row, loc, length, noise) in np.ndenumerate(start_ridges):
        # delete out of range, too low snr or too low length
        if not (peak_range[0] <= wa.time[loc] <= peak_range[1]) or length < min_length_percentage*len(scales) or ridge_max / noise < snr : 
            delete_ridge(ridge, row, col)
            start_ridges['row'][ind] = -1
            continue
            
    start_ridges = start_ridges[start_ridges['row'] != -1]
    
    #prev_max_row = start_ridges['max_row'].copy()
    #prev_max_loc = start_ridges['loc'].copy()
    # find the maximum row that does not include the asymmetric behavior
    scale_row = np.zeros(shape=start_ridges.shape, dtype=np.int32)
    scale_row_strength = np.zeros(shape=start_ridges.shape, dtype=np.int32)
    for ind, (row, col, ridge_max, max_row, loc, length, noise) in np.ndenumerate(start_ridges):
        scale_row[ind] = 0

        while ridge['from'][row, col] != -1:
            if abs(wa.time[loc] - wa.time[col]) < 10*width and not np.isclose(ridge['max'][row, col], ridge_max):
                scale_row[ind] = row
                scale_row_strength[ind] = ridge['power'][row, col]
                break
            
            col = ridge['from'][row, col]
            row = row-1    
          
        #start_ridges[ind]['max_row'] = scale_row[ind]
            
    # delete all that do not have expected range by estimating their uncertainty behaviour    
    coeff = nppoly.polyfit(np.sqrt(wa.time[start_ridges['loc']]),scales[scale_row],[1],w=start_ridges['max'])
    print(coeff)
    y_scale_fit = coeff[1]*np.sqrt(wa.time)+coeff[0]
   
    # find the power at the nearest scale level and compare to snr
    for ind, (row, col, ridge_max, max_row, loc, length, noise) in np.ndenumerate(start_ridges):      
        exp_scale = y_scale_fit[loc]
        
        strength = 0
        trow = row
        tcol = col
        while ridge['from'][trow, tcol] != -1:
            if scales[trow] < exp_scale:
                if trow != row or exp_scale > max_scale: strength = power[trow,tcol]
                break
            
            tcol = ridge['from'][trow, tcol]
            trow = trow-1    
            
        print(wa.time[loc], exp_scale, ridge_max, strength, noise) #, length, scales[trow], trow)
       
        if strength/noise < snr:
            # delete if not significant at the expected scale level
            delete_ridge(ridge, row, col)
            start_ridges['row'][ind] = -1
        elif strength > scale_row_strength[ind]:
            # optimize if previous scale estimation step was too aggressive
            start_ridges[ind]['max_row'] = trow
            start_ridges[ind]['loc'] = tcol
        else:
            #else just accept the previous estimate
            start_ridges[ind]['max_row'] = scale_row[ind]

    start_ridges = start_ridges[start_ridges['row'] != -1]    
   
    # delete not well separated peaks
    max_ridges = np.sort(start_ridges, order="max")[::-1]
    for rdg in max_ridges:
        bef, idx, aft = np.searchsorted(start_ridges['loc'], [rdg['loc']-peak_separation/width, rdg['loc'], rdg['loc']+peak_separation/width])
        
        if bef == aft or idx == start_ridges.shape[0] or start_ridges[idx]['loc'] != rdg['loc']: continue
        start_ridges['row'][bef:idx] = -1
        start_ridges['row'][idx+1:aft] = -1
        start_ridges = start_ridges[start_ridges['row'] != -1]
   
    return wa.time, wa.scales, ridge, start_ridges
    
def find_correlation_lines(power, width):
    ridge, start_ridges = find_ridge_lines(power, width, 0, 0.01, gap_thresh=1)
       
    # filter them..
    num_points = power[0,:].shape[0]
    ridge['max'] = 0
    for ind, (row, col, ridge_max, max_row, loc, length) in np.ndenumerate(start_ridges):                                    
        max_loc = -1
        
        if length < 50:
            delete_ridge(ridge, row, col)
            start_ridges[ind]['row'] = -1
            continue
        
        if np.isclose(ridge_max, power[row,col]):
            ridge['max'][row,col] = 1
            max_loc = row
            
        chk_len = 1
        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1
            
            if np.isclose(ridge_max, power[row,col]):
                max_loc = row
                ridge['max'][row,col] = 1
            else:
                ridge['max'][row,col] = 1
                
            chk_len = chk_len+1
                    
    start_ridges = start_ridges[start_ridges['row'] != -1]
    start_ridges = np.sort(start_ridges, order="length")[::-1]
    
    return ridge, start_ridges
    
