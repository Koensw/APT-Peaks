import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import matplotlib.pyplot as plt

from read import read_pos
from prepare import bin_data, limit_bins, zero_extend
from algs import estimate_baseline, find_peaks

from scipy.signal import gaussian, savgol_filter, convolve

import sortedcontainers

def prepare():
    print("Reading data...");
    data = read_pos('R62_01328-v07.pos', 200000000)
    
    print("Binning data...");
    bins, width = bin_data(data['m'], 0.01)
    bins = zero_extend(bins, width)
    
    print("Baseline removal...")
    baseline = estimate_baseline(bins, width, window=25, max_baseline=100)
    bins['height'] = bins['height']-baseline
    
    print("Smoothing...")
    gauss = gaussian(51, std=15, sym=True)
    gauss /= sum(gauss)
    bins['height'] = np.convolve(bins['height'], gauss, mode='same')
    
    return bins

def run(bins):
    plt.clf()
    #bins['height'] = savgol_filter(bins['height'], 101, 7)
    
    print("Finding peaks...");
    peaks = find_peaks(bins, min_peak_size=5, peak_diff=0.5)
    print(peaks)
    
    plt.plot(bins['edge']+width/2, bins['height'])
    bins['height'] = bins['height']+baseline
    plt.plot(bins['edge']+width/2, bins['height'], color='r')
    #plt.plot(bins['edge']+width/2, baseline)
    for el in peaks:
        plt.axvline(el['edge']+width/2, color='g')
    plt.yscale('log')
        
    plt.xlim(0, 200)
    #plt.ylim(0, peaks[0][0])
    axes = plt.gca()
    axes.set_ylim(ymin = 1)
    plt.show()

    
if __name__ == "__main__":
    bins = prepare()
    run(bins)