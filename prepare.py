""" 
Functions to prepare data for further processing
"""

import numpy as np
import math

"""
Create histogram of data skipping all bins with zero height

data: list of masses
accuracy: two times the width of the bins (maximum distance between a mass and its bin center)
"""
def bin_data(data, accuracy=0.1):
    min_element = min(data)
        
    # count the height of all bins
    idxs = np.floor((data-min_element)/(accuracy/2)).astype(int)
    vals, counts = np.unique(idxs, return_counts=True)
    edges = min_element+vals*(accuracy/2);
    
    # create data type containing a height and edge for all bins
    dt = np.dtype([('height', np.int64), ('edge', np.float64)])
    return np.rec.fromarrays([counts, edges], dt), accuracy/2
    
"""
Cap bins that fall outside a range

bins: list of bins returned by bin_data
min: minimum edge for a bin
max: maximum edge for a bin
"""
def cap_bins(bins, mini=-np.inf, maxi=np.inf):
    mask = mini <= bins['edge']
    mask &= bins['edge'] <= maxi
    return bins[mask]
    
"""
Make the bin data linear by filling up all the zero values

bins: list of bins returned by bin_data
width: width of bins
"""
def zero_extend(bins, width):
    ext_bins = np.zeros((int((bins[-1]['edge']-bins[0]['edge'])/width+1)+1,), dtype = bins.dtype)
    
    j = 0
    start = bins[0]['edge']
    for i, el in np.ndenumerate(bins):
        while (not math.isclose(start+j*width, el['edge'])) and start+j*width < el['edge']:
            ext_bins[j]['edge'] = start+j*width
            j = j+1
        
        # print(start+j*width, el['edge'])
        ext_bins[j] = el
        j = j+1
    
    return ext_bins[:j]
        
def limit_bins(bins, amount):
    idxs = np.argpartition(bins, amount, order='height')[-amount:]
    idxs.sort()
    return bins[idxs]
