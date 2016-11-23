"""
Collection of utilities to extend matplotlib for plotting 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from scipy.stats import norm

"""
Square root scale, particularly useful for plotting correlation histograms and spectra in (arbitrary) time units
"""
class SqrtScale(mscale.LinearScale):
    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        mscale.LinearScale.__init__(self, axis, **kwargs)

    def get_transform(self):
        return self.SqrtTransform()
        
    class SqrtTransform(mtransforms.Transform):
        def __init__(self):
           mtransforms.Transform.__init__(self)
           pass 
            
        def transform_non_affine(self, a):
            return np.sqrt(a)

        def inverted(self):
            return SqrtScale.InvertedSqrtTransform()

    class InvertedSqrtTransform(mtransforms.Transform):
        def __init__(self):
            mtransforms.Transform.__init__(self)
            pass

        def transform_non_affine(self, a):
            return a**2

        def inverted(self):
            return SqrtScale.SqrtTransform(self.lower, self.upper)         

"""
Set figsize for plots to include in LaTex (correct size of labels)

Based on http://bkanuka.com/articles/native-latex-plots/
"""
def figsize(scale): 
    fig_width_pt = 359.01668                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = 1.5*(np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size
            
"""
Create a new figure to include in LaTex

Based on http://bkanuka.com/articles/native-latex-plots/
"""
def newfig(num, width = 0.9):
    plt.close(num)
    fig = plt.figure(figsize=figsize(width), num=num)
    return fig

"""
Save a figure in PNG format (and PDF if provided)

Based on http://bkanuka.com/articles/native-latex-plots/
"""
def savefig(filename, pdf = False):
    if pdf: plt.savefig('{}.pdf'.format(filename))
    plt.savefig('{}.png'.format(filename), dpi=300)

"""
Set some optimized plotting parameters

Based on http://bkanuka.com/articles/native-latex-plots/
"""
def plot_init():
    import numpy as np
    import matplotlib as mpl

    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
        'figure.autolayout': True,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_with_latex)

    import matplotlib.pyplot as plt

    mscale.register_scale(SqrtScale)