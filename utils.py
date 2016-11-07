"""
Undocumented collection of utilities 
""" 

from .peaks import *

""" 
Find correlation lines using the ridge lines algorithm on a modified correlation histogram
"""
def find_correlation_lines(cwt, width):
    ridge, peak_info = find_ridge_lines(cwt, width, 0, 0.01, gap_thresh=1)
       
    # filter them..
    num_points = cwt[0,:].shape[0]
    ridge['max'] = 0
    for ind, (row, col, ridge_max, max_row, loc, length, noise) in np.ndenumerate(peak_info):                                    
        max_loc = -1
        
        if length < 50:
            delete_ridge(ridge, row, col)
            peak_info[ind]['row'] = -1
            continue
        
        if np.isclose(ridge_max, cwt[row,col]):
            ridge['max'][row,col] = 1
            max_loc = row
            
        chk_len = 1
        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1
            
            if np.isclose(ridge_max, cwt[row,col]):
                max_loc = row
                ridge['max'][row,col] = 1
            else:
                ridge['max'][row,col] = 1
                
            chk_len = chk_len+1
                    
    peak_info = peak_info[peak_info['row'] != -1]
    peak_info = np.sort(peak_info, order="length")[::-1]
    
    return ridge, peak_info
    