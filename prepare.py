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
    # find the minimum element as first edge index
    min_element = min(data)
    
    # divides the data in bins
    idxs = np.floor((data-min_element)/(accuracy/2)).astype(int)
    # count the height of all bins 
    vals, counts = np.unique(idxs, return_counts=True)
    # compute the starting edges
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
    # define a mask of true values for all values in range
    mask = mini <= bins['edge']
    mask &= bins['edge'] <= maxi
    
    # return only the value where the mask is true
    return bins[mask]
    
"""
Make the bin data linear by filling up all the zero values

bins: list of bins returned by bin_data
width: width of bins
"""
def zero_extend(bins, width):
    # create new list of extended_bins
    ext_bins = np.zeros((int((bins[-1]['edge']-bins[0]['edge'])/width+1)+1,), dtype = bins.dtype)
    
    # loop through the list filling the extended bins
    j = 0
    start = bins[0]['edge']
    for i, el in np.ndenumerate(bins):
        # add records for all empty bins
        while (not math.isclose(start+j*width, el['edge'])) and start+j*width < el['edge']:
            ext_bins[j]['edge'] = start+j*width
            j = j+1
        
        # add the already existent bins
        ext_bins[j] = el
        j = j+1
    
    return ext_bins[:j]
        
"""
Limit the bins to the highest

bins: input list of bins returned by bin_data
amount: amount of highest bins to remain
"""
def limit_bins(bins, amount):
    idxs = np.argpartition(bins, amount, order='height')[-amount:]
    idxs.sort()
    return bins[idxs]
