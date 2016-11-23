import numpy as np
import matplotlib.pyplot as plt

from peaks import find_peaks_cwt
from prepare import bin_data, cap_bins, zero_extend
from read import read_epos

EPOS_FILE = "M:/NetBeansProjects/R62_00933/recons/recon-v01/default/R62_00933-v01.epos" 
CONST_GAP_BIN = 0.001
CONST_MIN_BIN = 0
CONST_MAX_BIN = 200

"""
FIXME: add debug flag

#debug help
import warnings
#numpy treat warnings as errors
np.seterr(all='raise')
#next 2 lines print every RuntimeWarning as errors
warnings.filterwarnings("ignore")
warnings.simplefilter("error",category=RuntimeWarning)
"""

"""
TODO: integrate plotting tools

import plot_tools 
plot_tools.plot_init()
"""

#READ (E)POS FILE
data = read_epos(EPOS_FILE)
#debug
print("Atoms read: ", len(data))

#makes a histogram for full mass range
#raw_bins is a np.array('edge','height')
raw_bins, width = bin_data(data['m'], CONST_GAP_BIN)
#cap histogram
raw_bins = cap_bins(raw_bins, CONST_MIN_BIN, CONST_MAX_BIN)
#creates edges for the empty bins (can be included in bin_data)
raw_bins = zero_extend(raw_bins, width)

#debug command
print("Used bins in the mass spectrum from", raw_bins[0]['edge'], "to", raw_bins[-1]['edge'])

# find peaks
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

# loop through peaks
for rdg in peak_info:
    #mark peaks with estimated width
    axarr[0].axvspan((mass[rdg['loc']]-scales[rdg['max_row']]), (mass[rdg['loc']]+scales[rdg['max_row']]), color='y')
    #print only a marker at each identified peak
    #axarr[0].axvline(mass[rdg['loc']], color='y')

#prints a scatter plot the estimated scale for every peak
axarr[1].scatter((mass[peak_info['loc']]), scales[peak_info['max_row']])

for element in peak_info:
    print("Peak positions:", mass[element['loc']], "Peak strength:", element['max'])

plt.show()