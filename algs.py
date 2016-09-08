import numpy as np
import matplotlib.pyplot as plt

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
        
