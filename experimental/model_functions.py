"""
Collection of functions to model APT peaks

Experimental and mostly undocumenteds
"""

from scipy import signal
import numpy as np

"""
Gauss function 
"""
def gauss_func(m, S):
    c = 1/(np.sqrt(2*np.pi*S**2))
    with np.errstate(all="ignore"):
        f = np.exp(-1/2*(m/S)**2)
    return c*f

"""
Standard APT model using Arrhenius law and approximated temperature model
"""
def apt_model(t, t_0, B, C):
    #B = abs(B)
    #C = abs(C)
    if B < 0 or C < 0: return 0
    
    valid_range = np.inf
    
    y1 = np.zeros(np.count_nonzero(t < t_0))
    
    norm = (B**2*C)/(2*(1+B))*np.exp(B) #(B**2*C)/(2*(1+B))*np.exp(B)#(B**4*C**2)/(8*(3+B*(3+B)))*np.exp(B)
    mask = (t_0<=t) & (t<=t_0+valid_range)
    y2 = norm*np.exp(-B*np.sqrt(1+C*(t[mask]-t_0))) # np.sqrt(m[mask])-np.sqrt(m_0) # np.sqrt(m[mask]-m_0)

    y3 = np.zeros(np.count_nonzero(t > t_0+valid_range))
    
    y = np.concatenate((y1, y2, y3))
    return y

"""
Logarithmic APT model using Arrhenius law and approximated temperature model
"""    
# TODO SWITCH ALL TO TIME UNITS INSTEAD
def apt_log_model(m, m_0, B, C):
    B = abs(B)
    C = abs(C)
    
    y1 = np.zeros(np.count_nonzero(m <= m_0))
    norm = np.log((B**2*C)/(2*(1+B)))*B#(B**4*C**2)/(8*(3+B*(3+B)))*np.exp(B)
    y2 = norm*(-B*np.sqrt(1+C*(np.sqrt(m[m>m_0])-np.sqrt(m_0)))) # np.sqrt(m[m>=m_0])-np.sqrt(m_0) # np.sqrt(m[m>=m_0]-m_0)

    y = np.concatenate((y1, y2))
    return y    
    
"""
APT model convolved with a gauss function to model uncertainty
"""
def uncertain_apt_model(m, m_0, S, B, C):
    g = gauss_func(m-m[len(m)/2]+1e-8, S)
    a = apt_model(m, m_0, B, C)
    c = signal.fftconvolve(a, g, mode="same")*(m[1]-m[0])
    
    #print(np.argmax(a)-np.argmax(c))
    return np.roll(c, -1)

"""
Alternative APT model
"""
# FIXME: this does not work
def nonlinear_apt_model(m, m_0, A1, B1):
    t = np.sqrt(m[m>=m_0]-m_0)
    y1 = np.zeros(np.count_nonzero(m < m_0))
    y2 = A1**(np.exp(-B1*t))+A1**(B1/(B2-B1)*(np.exp(-B1*t)-np.exp(-B2*t)))*A2**np.exp(-B2*t)
    
    y = np.concatenate((y1, y2))
    return y

"""
Alternative APT model convolved with a gauss function for uncertainty
"""
def uncertain_nonlinear_apt_model(m, m_0, S, A1, B1):
    g = gauss_func(m-m[len(m)/2]+1e-8, S)
    a = nonlinear_apt_model(m, m_0, A1, B1)
    return signal.fftconvolve(a, g, mode="same")*(m[1]-m[0])
    
"""
Simple APT logarithmic model with standard value from literature
"""    
# TODO: switch all to time units
def simple_log_model(m, A, B):
    return A-1/0.05*np.sqrt(1+B*np.sqrt(m))
    
"""
Custom model for APT (normalized)
"""
def custom_model(t, t_0, tm_func, model_func):
    y1 = np.zeros(np.count_nonzero(t-t_0 < tm_func[0]))
    y3 = np.zeros(np.count_nonzero(t-t_0 > tm_func[-1]))
    
    idxs = np.searchsorted(tm_func, t[(tm_func[0] <= t-t_0) & (t-t_0 <= tm_func[-1])]-t_0)
    y2 = model_func[idxs]
    
    y = np.concatenate((y1, y2, y3))
    #print(len(y), len(t))
    return y
