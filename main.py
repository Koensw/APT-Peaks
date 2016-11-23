#%matplotlib qt5
#%reload_ext autoreload
#%autoreload 2

#import sys, os
import numpy as np
import matplotlib.pyplot as plt

from peaks import find_peaks_cwt
from prepare import bin_data, cap_bins, zero_extend

import plot_tools 
#from plot_tools import newfig, savefig
plot_tools.plot_init()

# INIT
from read import read_epos
file = "M:/NetBeansProjects/R62_00933/recons/recon-v01/default/R62_00933-v01.epos" #"data/R62_02111-v05.epos" # "data/R62_02196-v01.epos"  #"data/R62_02196-v01.epos"
# "//psnnas.phys.tue.nl/APT/NetBeansProjects/R62_01936/recons/recon-v01/default/R62_01936-v01.epos" 
# "data/R62_02196-v01.epos" "data/R62_02111-v05.epos" #"data/R62_01936-v01.epos" #"R62_01328-v07.pos"

#read epos file
data = read_epos(file)
#debug
print("Atoms read:",len(data))

#separate single from multi hit data
"""
data_single = data[data['Mhit'] == 1]
data_multi = data[data['Mhit'] != 1]
"""

#makes a histogram for full mass range
#raw_bins is a np.array('edge','height')
raw_bins, width = bin_data(data['m'], 0.001)
#cap histogram
CONST_MIN_BIN = 0
CONST_MAX_BIN = 200
raw_bins = cap_bins(raw_bins, CONST_MIN_BIN, CONST_MAX_BIN)
#creates edges for the empty bins (can be included in bin_data)
raw_bins = zero_extend(raw_bins, width)

#debug command
print("Used bins in the mass spectrum from",raw_bins[0]['edge'],"to",raw_bins[-1]['edge'])
# FIND PEAKS

#debug help
import warnings
#numpy treat warnings as errors
np.seterr(all='raise')
#next 2 line print every RuntimeWarning as errors
warnings.filterwarnings("ignore")
warnings.simplefilter("error",category=RuntimeWarning)

# mass = 1D list of mass centers = raw_bin edges + bin/2
# scales = 1D list of cwt scales
# ridge = contains cwt data for each mass and scale point (see peaks.py for details)
# peak_info = contains cwt info for the retrieved peaks (see peaks.py for details)
mass, scales, ridge, peak_info = find_peaks_cwt(raw_bins, width, 2.0,
                                                   peak_separation=0,
                                                   min_length_percentage = 0.4, noise_window = 1)

# opens figure 1
plt.figure(1)
# clear figure 1
plt.clf()
# f is pointer to the figure, axarr is pointer to the 2 subfigures
f, axarr = plt.subplots(2, sharex=True, num=1)
# plots the mass spectrum on a log scale into the first subfigure
axarr[0].plot(mass, raw_bins['height'], color='b')
axarr[0].set_yscale('log')

for rdg in peak_info:
    #mark peaks with estimated width
    axarr[0].axvspan((mass[rdg['loc']]-scales[rdg['max_row']]), (mass[rdg['loc']]+scales[rdg['max_row']]), color='y')
    #print only a marker at each identified peak
    #axarr[0].axvline(mass[rdg['loc']], color='y')

#prints a scatter plot the estimated scale for every peak
axarr[1].scatter((mass[peak_info['loc']]), scales[peak_info['max_row']])

for element in peak_info:
    print("Peak positions:", mass[element['loc']],"Peak strength:",element['max'])

plt.show()