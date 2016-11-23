"""
Collection of algorithms for peak detection
"""

import numpy as np
from numpy.polynomial import polynomial as nppoly
from scipy import signal #multiple signal functions from scipy

from wavelets import WaveletAnalysis, Ricker

import sortedcontainers

from .prepare import bin_data, cap_bins, zero_extend

def find_local_maxima(height, window_size, eps=10**-6,
                      initialization_width=10, min_peak_height=-np.inf):
    """
    Return 1D list of the indexes of the local maxima over a certain window size

    Args:
        (REQUIRED)
        height: input heights to find maxima on (1D list)
        window_size: length of the window to find local maxima on (integer normalized to bin units)

        (OPTIONAL)
        eps: smallest float that should be uniquely identified
             (precision due to conversion to integer)
        initialization_width: optimization parameter to pre-search maxima
            (simplifies spectra larger than 10 bins by looking for all
             maxima in each 10 bin before, scaling is always done to limit repercussions)
        min_peak_height: minimum height of local maxima to consider relevant

    Returns:
        List of indexes in the height array indicating the peaks
    """

    # initialize a balanced binary search tree set that support
    # efficient insertion and retrieval of the maxima
    bst = sortedcontainers.SortedSet()

    # initialize mask of final peak locations (array of FALSE with same length as height)
    mask = np.zeros(shape=height.shape, dtype=(np.bool))

    # minimize the peaks to search for by pre-searching maxima at a lower window size
    # NOTE: this algorithm scales with the window size and is
    # therefore not applicable at large window sizes
    # line picks max in every 10 bins -> gives list of maxima in all dimenions
    pos_peaks = signal.argrelmax(height, order=min(initialization_width, max(1, window_size//2)))[0]

    # loop through the remaining candidate peaks
    cnt_el = None #NULL pointer like
    bef = 0
    for pk_idx in pos_peaks:
        # convert the peak height to an integer using the input precision
        htc = height[pk_idx]
        if np.isnan(htc):
            htc = 0 # makes sure logarithmic spectra "work"
        htc = int(htc / eps)

        # add the current peak to the set -> bst will automatically sort itself after adding
        bst.add((htc, pk_idx))

        # remove all peaks that now fall outside the window
        while pk_idx - pos_peaks[bef] > window_size:
            # compute the peak height of the peaks that are removed
            inp = pos_peaks[bef]
            htp = height[inp]
            if np.isnan(htp):
                htp = 0
            htp = int(htp / eps)

            # check if this peak was a maximum over the whole window (and therefore a local maxima)
            if cnt_el == (htp, inp) and cnt_el[0] > min_peak_height / eps:
                mask[inp] = True

            # remove the peak and continue to the next one
            bst.remove((htp, inp))
            bef = bef+1

        # save the newly added peak if it is the new maximum
        if bst[-1][0] == htc:
            cnt_el = bst[-1]

        # remove the current maximum candidate if it is not the maximum anymore
        if cnt_el and not bst[-1] == cnt_el:
            cnt_el = None

    # add the final local maximum if it exists
    if cnt_el and cnt_el[0] > min_peak_height / eps:
        mask[cnt_el[1]] = True

    # return the location of the peaks
    return np.nonzero(mask)[0]

def delete_ridge(ridge, row, col):
    """
    Delete a ridge line given it starting location

    Args:
        ridge: 2D array that store the ridge information
        row: starting row of the ridge
        col: starting column of the ridge
    """

    # set all values of the ridge array to the empty value
    ridge['loc'][row, col] = -1
    ridge['max'][row, col] = 0
    ridge['length'][row, col] = 0

    while ridge['from'][row, col] != -1:
        col = ridge['from'][row, col]
        row = row-1

        ridge['loc'][row, col] = -1
        ridge['max'][row, col] = 0
        ridge['length'][row, col] = 0

#TODO: gap_tresh should be in real differences or everything should be in bin differences
def find_ridge_lines(cwt, width, left_window, right_window, gap_thresh=2,
                     noise_window=0.5, min_noise=0.1):
    """
    Find ridge lines and extract information from the wavelet transform coefficients

    Args:
        (REQUIRED)
        cwt: the calculated cwt coefficients
        width: the gap between histogram bins
            (used to convert real differences to discrete bin differences)
        right_window: right window sizes over all scales in real size differences
            (or scalar if the same window on all levels)
        left_window: left window sizes over all scales in real size differences
            (or scalar if the same window on all levels)

        (OPTIONAL)
        gap_thresh: threshold of allowed gap for levels where peaks are not present
            before saving it as new ridge line (depends on the amount of scales used)
        noise_window: window on real scale to compute the noise
        min_noise: minimum level of noise

    Returns:
        Tuple containing
        - 2D array of ridge information (defined below)
        - 1D list of possible peaks with related ridge (defined below)
    """

    # convert real window to discrete window over the bins
    left_window_list = np.array(left_window/width, dtype=np.float, ndmin=1, copy=False)
    right_window_list = np.array(right_window/width, dtype=np.float, ndmin=1, copy=False)

    # extend the window list to its correct size if a scalar input is provided
    left_window_list = np.lib.pad(left_window_list,
                                  (0, cwt.shape[0]-left_window_list.shape[0]), 'edge')
    right_window_list = np.lib.pad(right_window_list,
                                   (0, cwt.shape[0]-right_window_list.shape[0]), 'edge')

    # convert noise window to discrete window over the bins
    noise_window = int(noise_window/width)

    # define the stucture containing the ridge information and initialize it
    # the ridge structure contain a tuple with information for every combination of mass and scale
    # row is used to loop over the scales and column to loop over the masses
    dtype = np.dtype([('cwt', np.float),  # the CWT coefficients (copy of argument cwt)
                      ('peak', np.bool),    # boolean array indicating if the index is a maximum at the specific mass and scale
                      ('from', np.int64),   # indiciate the column on the previous row for this ridge line (or -1 if first column)
                      ('gap', np.int64),    # difference where the peak at (row-gap,col) was last matched
                      ('max', np.float),  # maximum CWT coefficient on the ridge line so far
                      ('loc', np.int64),    # mass location of the ridge line (the mass on the lowest scale level without optimisation)
                      ('length', np.int)])  # length of the ridge line

    # initialize array and set default values
    ridge = np.ndarray(shape=cwt.shape, dtype=dtype)
    ridge['length'] = 0
    ridge['peak'] = False
    ridge['gap'] = 0
    ridge['max'] = 0
    ridge['from'] = -1
    ridge['loc'] = -1
    ridge['cwt'] = cwt

    # initialize a structure that saves a list of peaks on earlier scale levels
    last_peaks = np.empty((0), dtype=np.int32)

    # define and initialize final list of possible peaks (with their starting point of the ridge)
    dtype = np.dtype([('row', np.int32),            # the row index in the list of ridges where the ridge line start (represents the index of the maximum scale level)
                      ('col', np.int32),            # the column index in the list of ridges where the ridge line start (represents the index of mass location at the maximum scale level)
                      ('max', np.float),          # maximum CWT coefficient on the associated ridge line (represent the strength of the peak)
                      ('max_row', np.int32),        # scale where the maximum CWT coefficient is located (related to the best-matching window size)
                      ('loc', np.int32),            # mass of the peak (the mass on the lowest scale level without optimisation)
                      ('length', np.int32),         # length of the ridge line associated with the peak
                      ('noise', np.float)])       # estimated strength of the noise around the peak
    peak_info = np.empty(cwt.shape[1], dtype=dtype)
    peak_info_idx = 0

    # enumerate over all scale levels
    for row, (left_window, right_window) in enumerate(np.c_[left_window_list, right_window_list]):
        # find the local maxima on the current level
        peaks = find_local_maxima(cwt[row, :], int(max(1, left_window+right_window)), min_peak_height=0)
        ridge['peak'][row, peaks] = True

        # loop over all previous maxima
        points = np.searchsorted(last_peaks, peaks)
        last_peaks_length = len(last_peaks)
        keep_mask = np.ones(last_peaks.shape, dtype=np.bool)
        for ind, ins_idx in np.ndenumerate(points):
            min_i = -1
            min_v = max(left_window, right_window)+1

            # find the nearest maxima on previous levels
            if ins_idx != 0 and peaks[ind]-left_window <= last_peaks[ins_idx-1] <= peaks[ind]+right_window and peaks[ind]-last_peaks[ins_idx-1] < min_v:
                min_v = peaks[ind]-last_peaks[ins_idx-1]
                min_i = ins_idx-1
            if ins_idx != last_peaks_length and peaks[ind]-left_window <= last_peaks[ins_idx] <= peaks[ind]+right_window and last_peaks[ins_idx]-peaks[ind] < min_v:
                min_v = last_peaks[ins_idx]-peaks[ind]
                min_i = ins_idx

            # check if current peak is in range of peak on previous level
            # TODO: improve use of gap for better matching
            if min_v <= max(left_window, right_window):
                # if in range then extend the earlier ridge line
                ridge['from'][row, peaks[ind]] = last_peaks[min_i]
                ridge['gap'][row, peaks[ind]] = 0
                ridge['max'][row, peaks[ind]] = max(cwt[row, peaks[ind]], ridge['max'][row-1, last_peaks[min_i]])
                ridge['length'][row, peaks[ind]] = ridge['length'][row-1, last_peaks[min_i]]+1
                ridge['loc'][row, peaks[ind]] = ridge['loc'][row-1, last_peaks[min_i]]

                # TODO: check if matching of the same earlier peak occurs and if this lead to significant errors
                keep_mask[min_i] = 0
            else:
                # if not in range then a new ridge line should be initialized
                ridge['from'][row, peaks[ind]] = -1
                ridge['gap'][row, peaks[ind]] = 0
                ridge['length'][row, peaks[ind]] = 1
                ridge['max'][row, peaks[ind]] = cwt[row, peaks[ind]]
                ridge['loc'][row, peaks[ind]] = peaks[ind]

        # update all non-matched ridge lines
        for i, idx in np.ndenumerate(last_peaks):
            if keep_mask[i] == 0:
                continue

            # extend all non-matched peaks to the next level
            ridge['gap'][row, idx] = ridge['gap'][row-1, idx]+1
            ridge['from'][row, idx] = idx
            ridge['max'][row, idx] = ridge['max'][row-1, idx]
            ridge['loc'][row, idx] = ridge['loc'][row-1, idx]
            ridge['length'][row, idx] = ridge['length'][row-1, idx]

            # remove the ridge lines from consideration if it exceeds the threshold of times where it is not matched
            if ridge['gap'][row, idx] > gap_thresh:
                peak_info[peak_info_idx]['row'] = row
                peak_info[peak_info_idx]['col'] = idx
                peak_info[peak_info_idx]['max'] = ridge['max'][row, idx]
                peak_info[peak_info_idx]['loc'] = ridge['loc'][row, idx]
                peak_info[peak_info_idx]['length'] = ridge['length'][row, idx]

                # double the size of the final list of ridges if it exceeds the current size
                peak_info_idx = peak_info_idx + 1
                if peak_info_idx >= peak_info.shape[0]:
                    peak_info = np.pad(peak_info, (0, peak_info.shape[0]), 'edge')
                keep_mask[i] = 0

        # remove the earlier peaks that are matched or passing the gap threshold
        last_peaks = last_peaks[keep_mask]

        # merge the newly found peaks
        last_peaks = np.concatenate((last_peaks, peaks))
        last_peaks.sort(kind='mergesort')

    # save the final ridge lines
    for i, idx in np.ndenumerate(last_peaks):
        peak_info[peak_info_idx]['row'] = cwt.shape[0]-1
        peak_info[peak_info_idx]['col'] = idx
        peak_info[peak_info_idx]['max'] = ridge['max'][-1, idx]
        peak_info[peak_info_idx]['loc'] = ridge['loc'][-1, idx]
        peak_info[peak_info_idx]['length'] = ridge['length'][-1, idx]
        peak_info_idx = peak_info_idx + 1

        # double the size of the final list of ridges if it exceeds the current size
        if peak_info_idx >= peak_info.shape[0]:
            peak_info = np.pad(peak_info, (0, peak_info.shape[0]), 'edge')

    # resize the list to contain all actual ridge lines
    peak_info = peak_info[:peak_info_idx]

    # back traverse the ridge lines to compute parameters
    noise_level = 0
    num_points = cwt[0, :].shape[0]
    for ind, (row, col, ridge_max, _, loc, _, _) in np.ndenumerate(peak_info):
        # find the row (the scale index) where the maximum is located
        if np.isclose(ridge_max, cwt[row, col]):
            peak_info[ind]['max_row'] = row
            peak_info[ind]['loc'] = col

        while ridge['from'][row, col] != -1:
            col = ridge['from'][row, col]
            row = row-1

            if np.isclose(ridge_max, cwt[row, col]):
                peak_info[ind]['max_row'] = row

        # set the approximated location of the peak to the location at the lowest scale level
        peak_info[ind]['loc'] = col

        # estimate noise using the lowest bin level over the defined window
        window_start = max(loc - noise_window, 0)
        window_end = min(loc + noise_window, num_points)
        peak_info[ind]['noise'] = max(min_noise, np.percentile(abs(cwt[noise_level, window_start:window_end]), 95))

    # sort the ridges on starting location and return them
    peak_info = np.sort(peak_info, order="loc")

    # return the 2D ridge array and the 1D list of peak and associated ridges
    return ridge, peak_info

def find_peaks_cwt(bins, width, snr, min_length_percentage=0.4,
                   peak_range=(0.5, np.inf), peak_separation=0,
                   gap_scale=0.05, gap_thresh=2, noise_window=1.0):
    """
    Find peaks in APT spectra through CWT, ridge lines and filtering

    Args:
        (REQUIRED)
        bins: mass histogram bins (need linear scaling!) (1D list)
        width: the width of the bins
        snr: required signal-to-noise ratio for significant peaks

        (OPTIONAL)
        min_length_percentage: minimum percentage of scales where the peak should exist
        peak_range: looks for peaks in this range (now from 0.5 to inf -> so peaks before 0.5 are removed)
        peak_separation: an optional distance between peaks (peaks below this threshold are merged)
        gap_scale: width between scales given in units of 2^width having width as linear scale with gap_scale separation
        gap_thresh: threshold of allowed gap for levels where peaks are not present before saving it as new ridge line (depends on the amount of scales used)
        noise_window: window on real scale to compute the noise

    Returns:
        Tuple with
        - list of masses
        - list of scales
        - 2D array of ridges (see find_ridge_lines)
        - 1D list of information on peaks (see find_ridge_lines)
    """

    # select the wavelet object to use - the ricker (mexican hat) wavelet gives good results
    wavelet = Ricker()

    # select the minimum and maximum applicable scale levels
    min_scale = np.log2(4*width)
    max_scale = np.log2(2) #this are 2 amu at the moment

    # compute the logarithm of the height as the CWT performs better here on typical spectra
    height_log = bins['height'].astype(np.float)
    height_log += 1 # add one to circumvent the problem of zero height
    height_log = np.log(height_log)

    # initialize the wavelet library
    wav = WaveletAnalysis(height_log, wavelet=wavelet, dt=width)
    wav.time = bins['edge']+width/2 # use mass to charge as the time scale
    scales = wav.scales = 2**(np.arange(min_scale, max_scale, gap_scale)) # build a set of scales
    # apply the continous wavelet transform and keep the real part (in theory there should be no imaginary part)
    cwt = np.real(wav.wavelet_transform)

    #ridge is 2D array of tuples where each tuple contains (see function above)
    #(cwt-intensity, local-max-yes-no, previous-point-on-line, scale-gap-to-previous-point, start-of-ridge, current-length-of-ridge)
    #peak_info 1D list of tuples where each tuple contains (see function above)
    #(scale-of-ridge-start(row), mass-of-ridge-start(col), start-of-ridge, max-length-of-ridge, max-intensity, scale-of-max-intensity)
    #called with: cwt-object (3D array scale, mass, intensities), bin-width, left and right window for ridge lines
    #scale levels that it skips before ridge line ends, window to check noise at lowest scale level)
    ridge, peak_info = find_ridge_lines(cwt, width, scales/2, scales/2, gap_thresh, noise_window)
    #TODO: limiting what ridge saves is saving a LOT OF MEMORY

    # correct maxima for snr and push them down deleting any that does not work
    for ind, (row, col, ridge_max, _, loc, length, noise) in np.ndenumerate(peak_info):
        # delete peaks that:
        # - are not in parameter peak_range or
        # - dont appear as a single line for enough scale or
        # - too low snr (noise comes from wavelet estimate)
        if not (peak_range[0] <= wav.time[loc] <= peak_range[1]) or length < min_length_percentage*len(scales) or ridge_max / noise < snr:
            delete_ridge(ridge, row, col)
            peak_info['row'][ind] = -1
            continue

    peak_info = peak_info[peak_info['row'] != -1]

    # find a guess of the maximum row that does not include the asymmetric behavior
    scale_row = np.zeros(shape=peak_info.shape, dtype=np.int32)
    scale_row_strength = np.zeros(shape=peak_info.shape, dtype=np.int32)
    for ind, (row, col, ridge_max, _, loc, length, noise) in np.ndenumerate(peak_info):
        #traversing down the ridge
        while ridge['from'][row, col] != -1:
            # stop at first peak that is near the start location and not above the original maximum
            if abs(wav.time[loc] - wav.time[col]) < 10*width and not np.isclose(ridge['max'][row, col], ridge_max):
                scale_row[ind] = row
                scale_row_strength[ind] = ridge['cwt'][row, col]
                break

            col = ridge['from'][row, col]
            row = row-1

    # delete all that do not have expected range by estimating their uncertainty behaviour
    # coeff is tuole of 2 (steepness of a line fit, const. line fit=0)
    coeff = nppoly.polyfit(np.sqrt(wav.time[peak_info['loc']]), scales[scale_row], [1], w=peak_info['max'])

    #resulting linear fit
    #print(coeff)
    y_scale_fit = coeff[1]*np.sqrt(wav.time)+coeff[0]

    # find the cwt coefficient at the nearest scale level and compare to snr
    for ind, (row, col, ridge_max, _, loc, length, noise) in np.ndenumerate(peak_info):
        exp_scale = y_scale_fit[loc]

        strength = 0
        trow = row
        tcol = col
        #if the expected row is never reached (peak only exists on higher scales) -> strength stays 0
        #after this loop peaks that dont exist at the expected strength are 0 and peaks that exist have their intensity in strength
        while ridge['from'][trow, tcol] != -1:
            if scales[trow] < exp_scale:
                #if the first row on the traverse down is below the scale -> strength stays 0
                #if exp_scale > max_scale than the fit is bad and we just keep everything
                if trow != row or exp_scale > max_scale:
                    strength = cwt[trow, tcol]
                break

            tcol = ridge['from'][trow, tcol]
            trow = trow-1

        #print(wav.time[loc], exp_scale, ridge_max, strength, noise) #, length, scales[trow], trow)

        #kicks out peaks that dont have enough signal at the expected scale level at this point in the mass spectrum
        if strength/noise < snr:
            # delete if not significant at the expected scale level
            delete_ridge(ridge, row, col)
            peak_info['row'][ind] = -1
        elif strength > scale_row_strength[ind]:
            # optimize if previous scale estimation step was too aggressive
            peak_info[ind]['max_row'] = trow
            peak_info[ind]['loc'] = tcol
        else:
            #else just accept the previous estimate as actual max row
            peak_info[ind]['max_row'] = scale_row[ind]

    # remove all filtered peaks
    peak_info = peak_info[peak_info['row'] != -1]

    # delete not well separated peaks (optional step)
    max_ridges = np.sort(peak_info, order="max")[::-1]
    for rdg in max_ridges:
        # find peaks near others
        bef, idx, aft = np.searchsorted(peak_info['loc'], [rdg['loc']-peak_separation/width, rdg['loc'], rdg['loc']+peak_separation/width])

        # remove the peak if larger is inside the separation range
        if bef == aft or idx == peak_info.shape[0] or peak_info[idx]['loc'] != rdg['loc']:
            continue

        peak_info['row'][bef:idx] = -1
        peak_info['row'][idx+1:aft] = -1
        peak_info = peak_info[peak_info['row'] != -1]

    # return the list of masses (center of the bins), scales (defined above), ridge information and the peak information including their related ridge (from the find_ridge_lines function)
    return wav.time, wav.scales, ridge, peak_info

def find_multi_peaks(data, mass, scales, peak_info, snr, significant_peak_strength, simultaneous_uncertainty=0.75, **cwt_options):
    """
    Optimize peak info by finding peaks in the multihit spectrum

    Args:
        (REQUIRED)
        data: the mass and a number indicationg the multihit returned by read_epos
        mass: range of masses used in producing peak_info
        scales: range of scales ued while producing peak_info
        peak_info: ridge list of earlier peaks
        snr: required signal-to-noise ratio for significant peaks
        significant_peak_strenghth: strength for which a peak is expected to have significant influence on the next peak and its simultaneous evaporation has to be taken into account

        (OPTIONAL)
        simultaneous_uncertainty: uncertainty in mass of a simultaneous evaporation
        **cwt_options: other options to pass to cwt algorithm

    Returns:
        Tuple with
        - list of masses
        - list of scales
        - bins produced containing the approximated peak shape centered at zero
        - 2D array of ridges (see find_ridge_lines)
        - 1D list of information on peaks (see find_ridge_lines)
    """
    # use the bin width and peak range of earlier step
    bin_width = mass[1]-mass[0]
    peak_range = (mass[0], mass[-1])

    # sort on location (if not already done before)
    peak_info = np.sort(peak_info, order=['loc'])

    # build a list of ranges for every peak where every other peak in the multihit should be ignored
    peak_ignore = np.empty(shape=(peak_info.shape[0], peak_info.shape[0]*peak_info.shape[0]), dtype=np.float)
    peak_ignore[:] = -1
    peak_ignore_cnt = np.zeros(shape=peak_info.shape[0], dtype=np.int32)

    # iterate through all peak combinations
    for ind1, peak1 in enumerate(peak_info):
        for ind2, peak2 in enumerate(peak_info):
            # ignore all peaks if not both are significant
            # also ignore combinations of the same peak as these will never add extra wrong peak locations
            if ind1 == ind2 or peak1['max'] < significant_peak_strength or peak2['max'] < significant_peak_strength:
                continue

            # compute the constant for the line describing the diagonal simultaneous evaporation
            srx = mass[peak1['loc']]
            sry = mass[peak2['loc']]
            src = np.sqrt(srx)-np.sqrt(sry)

            # loop through all peaks
            for ind, peak in enumerate(peak_info):
                loc = peak['loc']

                # skip all peaks before the constant is passed
                if np.sqrt(mass[loc]) < src:
                    continue

                # find the matching index for this peak where values around should be ignored
                sli = np.searchsorted(mass, (np.sqrt(mass[loc])-src)**2)

                # also ignore if the last intersection is passed
                if sli == len(mass):
                    continue

                # add the center point to the ignore list
                peak_ignore[ind][peak_ignore_cnt[ind]] = mass[sli]
                peak_ignore_cnt[ind] = peak_ignore_cnt[ind] + 1

    # sort the list of ranges where peaks should be ignored on location
    for i in range(0, len(peak_info)):
        peak_ignore[i] = np.sort(peak_ignore[i])

    # remove all singlehits
    data_multi = data[data['Mhit'] != 1]

    # create a mask that will contain the valid filtered masses
    mask = np.ndarray(shape=data_multi.shape, dtype=np.bool)
    mask[:] = False

    # loop through the data
    for i, elm in enumerate(data_multi):
        # only start from the actual multi hit
        if elm['Mhit'] == 0:
            continue

        # loop through this multi hit
        for j in range(0, elm['Mhit']):
            # ignore ions outside the peak range
            if not peak_range[0] < data_multi[i+j]['m'] < peak_range[1]:
                continue

            # find the nearest peak
            pk_idx = np.searchsorted(mass[peak_info['loc']], data_multi[i+j]['m'])
            if not ((pk_idx != len(peak_info['loc']) and abs(data_multi[i+j]['m']-mass[peak_info['loc'][pk_idx]]) <= scales[peak_info['max_row'][pk_idx]]) or
                    (pk_idx and abs(data_multi[i+j]['m']-mass[peak_info['loc'][pk_idx-1]]) <= scales[peak_info['max_row'][pk_idx-1]])):
                continue

            # find the nearest peak
            if pk_idx == len(peak_info['loc']) or (pk_idx and data_multi[i+j]['m']-mass[peak_info['loc'][pk_idx-1]] < mass[peak_info['loc'][pk_idx]]-data_multi[i+j]['m']):
                pk_idx = pk_idx-1

            # loop to all the multi hit combinations
            for k in range(j+1, elm['Mhit']):
                if not peak_range[0] < data_multi[i+k]['m'] < peak_range[1]:
                    continue

                # find the location in the ignore list
                ins_idx = np.searchsorted(peak_ignore[pk_idx], data_multi[i+k]['m'])

                # find the distance to the nearest peak in the ignore list
                min_sz = np.inf
                if ins_idx:
                    min_sz = min(min_sz, data_multi[i+k]['m']-peak_ignore[pk_idx][ins_idx-1])
                if ins_idx != len(peak_ignore[pk_idx]):
                    min_sz = min(min_sz, peak_ignore[pk_idx][ins_idx]-data_multi[i+k]['m'])

                # ignore this combination if is too near a wrong location
                if min_sz < simultaneous_uncertainty:
                    continue

                # this measurement is within a valid combination thus assign it to the valid list
                mask[i+k] = True

    # create a histogram of the filtered masses
    mult_bins, width = bin_data(data_multi[mask]['m'], bin_width)
    mult_bins = cap_bins(mult_bins, peak_range[0], peak_range[1])
    mult_bins = zero_extend(mult_bins, width)

    # find the peaks using the algorithm
    new_mass, new_scales, ridge, new_peak_info = find_peaks_cwt(mult_bins, width, snr, *cwt_options)

    # get the non intersecting peaks
    new_peak_info = get_non_intersecting_peaks(mass, new_mass, scales, peak_info, new_peak_info)

    return new_mass, new_scales, mult_bins, ridge, new_peak_info

def get_non_intersecting_peaks(base_mass, new_mass, base_scales, base_peak_info, new_peak_info):
    """
    Find all the peaks in the second list that are not in range of the first one

    Args:
        base_mass: mass units of the base bins
        new_mass: mass units of the bins to check for intersection
        base_scales: scale units for the base bins
        base_peak_info: peak info for the base bins from find_peaks_cwt
        new_peak_info: peak info for the bins to check for intersection

    Returns:
        List of information from non-intersecting peak
    """

    # loop through new peaks
    for ind, peak in enumerate(new_peak_info):
        # find location of nearest peak in the original list
        ins_idx = np.searchsorted(base_mass[base_peak_info['loc']], new_mass[peak['loc']])
        min_sz = np.inf
        if ins_idx != len(base_peak_info) and base_mass[base_peak_info['loc'][ins_idx]] - new_mass[peak['loc']] < min_sz:
            min_sz = base_mass[base_peak_info['loc'][ins_idx]] - new_mass[peak['loc']]
        if ins_idx and new_mass[peak['loc']] - base_mass[base_peak_info['loc'][ins_idx-1]] < min_sz:
            min_sz = new_mass[peak['loc']] - base_mass[base_peak_info['loc'][ins_idx-1]]
            ins_idx = ins_idx-1

        # remove all peaks that are in the other list
        if min_sz < base_scales[base_peak_info['max_row'][ins_idx]]:
            new_peak_info[ind]['row'] = -1

    # return non intersecting peaks
    new_peak_info = new_peak_info[new_peak_info['row'] != -1]
    return new_peak_info
