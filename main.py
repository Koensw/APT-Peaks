import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import matplotlib.pyplot as plt

from .read import read_pos
from .prepare import bin_data, limit_bins, zero_extend
from .algs import estimate_baseline, find_peaks

from scipy.signal import gaussian, savgol_filter, convolve

import sortedcontainers

def read(file):
    print("Reading data...");
    data = read_pos(file)
    
    print("Binning data...");
    bins, width = bin_data(data['m'], 0.01)
    bins = zero_extend(bins, width)
    
    return bins, width

def prepare(bins, width):    
    print("Baseline removal...")
    baseline = estimate_baseline(bins, width, window=25, max_baseline=100)
    bins['height'] = bins['height']-baseline
    
    print("Smoothing...")
    gauss = gaussian(51, std=15, sym=True)
    gauss /= sum(gauss)
    bins['height'] = np.convolve(bins['height'], gauss, mode='same')
    
    #bins['height'] = savgol_filter(bins['height'], 101, 7)
    
    return bins

def run(bins, width):
    print("Finding peaks...");
    peaks = find_peaks(bins, min_peak_size=5, peak_diff=0.5)
    print(peaks)
    
    return peaks

def plot(bins, raw_bins, peaks, width):
    plt.clf()
    plt.plot(raw_bins['edge']+width/2, raw_bins['height'], color='r')
    plt.plot(bins['edge']+width/2, bins['height'], color='b')
    
    for el in peaks:
        plt.axvline(el['edge']+width/2, color='g')
    plt.yscale('log')
        
    plt.xlim(0, 200)
    #plt.ylim(0, peaks[0][0])
    axes = plt.gca()
    axes.set_ylim(ymin = 1)
    plt.show()
    
if __name__ == "__main__":
    file = "R62_01328-v07.pos"

    raw_bins, width = read(file)
    bins = prepare(raw_bins.copy(), width)
    
    peaks = run(bins, width)
    plot(bins, raw_bins, peaks, width)
    
    