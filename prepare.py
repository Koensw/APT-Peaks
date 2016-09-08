import numpy as np
import math

def bin_data(data, accuracy=0.1):
    min_element = min(data)
        
    idxs = np.floor((data-min_element)/(accuracy/2)).astype(int)
    vals, counts = np.unique(idxs, return_counts=True)
    edges = min_element+vals*(accuracy/2);
    
    dt = np.dtype([('height', np.int64), ('edge', np.float64)])
    return np.rec.fromarrays([counts, edges], dt), accuracy/2
    
def zero_extend(bins, width):
    narr = np.zeros((int((bins[-1]['edge']-bins[0]['edge'])/width+1),), dtype = bins.dtype)
    
    j = 0
    start = bins[0]['edge']
    for i, el in np.ndenumerate(bins):
        while not math.isclose(start+j*width, el['edge']):
            narr[j]['edge'] = start+j*width
            j = j+1
        
       # print(start+j*width, el['edge'])
        narr[j] = el
        j = j+1
    
    return narr
        
def limit_bins(bins, amount):
    idxs = np.argpartition(bins, amount, order='height')[-amount:]
    idxs.sort()
    return bins[idxs]