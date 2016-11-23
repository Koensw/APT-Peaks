#%matplotlib qt5
#%reload_ext autoreload
#%autoreload 2

import sys, os
import apt_peaks as ap
from apt_peaks.main import *

import plot_tools 
from plot_tools import newfig, savefig
plot_tools.plot_init()

# INIT
from apt_peaks.read import read_epos
file = "data/R62_01936-v01.epos" #"data/R62_02111-v05.epos" # "data/R62_02196-v01.epos"  #"data/R62_02196-v01.epos"
# "//psnnas.phys.tue.nl/APT/NetBeansProjects/R62_01936/recons/recon-v01/default/R62_01936-v01.epos" 
# "data/R62_02196-v01.epos" "data/R62_02111-v05.epos" #"data/R62_01936-v01.epos" #"R62_01328-v07.pos"

data = read_epos(file)
data_single = data[data['Mhit'] == 1]
data_multi = data[data['Mhit'] != 1]

raw_bins, width = bin_data(data['m'], 0.001)
raw_bins = cap_bins(raw_bins, 0, 140)
raw_bins = zero_extend(raw_bins, width)

# FIND PEAKS
import warnings
from numpy.polynomial import polynomial as nppoly
from apt_peaks.peaks import find_peaks_cwt

np.seterr(all='raise')
warnings.filterwarnings("ignore")
warnings.simplefilter("error",category=RuntimeWarning)

mass, scales, ridge, start_ridges = find_peaks_cwt(raw_bins, width, 2.0,
                                                   peak_separation=0,
                                                   min_length_percentage = 0.4, noise_window = 1)
peak_info = start_ridges

plt.figure(1)
plt.clf()
f, axarr = plt.subplots(2, sharex=True, num=1)
axarr[0].plot((raw_bins['edge']+width/2), raw_bins['height'], color='b')
axarr[0].set_yscale('log')


"""
fsdfas


"""
peak_intervals = np.zeros(shape=(2*len(start_ridges),))
idx = 0
for rdg in start_ridges:
    if idx != 0 and mass[rdg['loc']]-scales[rdg['max_row']] < peak_intervals[idx-1]:
        peak_intervals[idx-1] = mass[rdg['loc']]+scales[rdg['max_row']]
    else:
        peak_intervals[idx] = mass[rdg['loc']]-scales[rdg['max_row']]
        peak_intervals[idx+1] = mass[rdg['loc']]+scales[rdg['max_row']]
        idx += 2
    
    axarr[0].axvspan((mass[rdg['loc']]-scales[rdg['max_row']]), (mass[rdg['loc']]+scales[rdg['max_row']]), color='y')
    #axarr[0].axvline(mass[rdg['loc']], color='y')

peak_intervals = peak_intervals[:idx]
    
ridge_log = ridge['max']
ridge_log[ridge_log < 1e-4] = np.nan
ridge_log = np.log(ridge_log)

axarr[1].scatter((mass[start_ridges['loc']]), scales[start_ridges['max_row']])
axarr[1].plot(mass, yfit)