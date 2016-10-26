import scipy
import numpy as np
import math

from wavelets import DOG, Ricker

# a Ricker function that gives an direct unbiased wavelet transform
class UnbiasedRicker(Ricker):        
    def time(self, t, s=1.0):
        return Ricker.time(self, t, s) #/ np.sqrt(s)


class HalfDOG(object):
    def __init__(self, m=2):
        """Initialise a Derivative of Gaussian wavelet of order `m`."""
        if m == 2:
            # value of C_d from TC98
            self.C_d = 3.541
        elif m == 6:
            self.C_d = 1.966
        else:
            pass
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return a Derivative of Gaussian wavelet,
        When m = 2, this is also known as the "Mexican hat", "Marr"
        or "Ricker" wavelet.
        It models the function::
            ``A d^m/dx^m exp(-x^2 / 2)``,
        where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5``
        and   ``x = t / s``.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        Parameters
        ----------
        t : float
            Time. If `s` is not specified, this can be used as the
            non-dimensional time t/s.
        s : scalar
            Width parameter of the wavelet.
        Returns
        -------
        out : float
            Value of the DOG wavelet at the given time
        Notes
        -----
        The derivative of the Gaussian has a polynomial representation:
        from http://en.wikipedia.org/wiki/Gaussian_function:
        "Mathematically, the derivatives of the Gaussian function can be
        represented using Hermite functions. The n-th derivative of the
        Gaussian is the Gaussian function itself multiplied by the n-th
        Hermite polynomial, up to scale."
        http://en.wikipedia.org/wiki/Hermite_polynomial
        Here, we want the 'probabilists' Hermite polynomial (He_n),
        which is computed by scipy.special.hermitenorm
        """
        x = t / s
        m = self.m

        # compute the Hermite polynomial (used to evaluate the
        # derivative of a Gaussian)
        He_n = scipy.special.hermitenorm(m)
        Un_n = lambda x: 0.5 * (np.sign(x) + 1);
        gamma = scipy.special.gamma

        const = (-1) ** (m + 1) / gamma(m + 0.5) ** .5
        function = np.sqrt(2) * Un_n(x) * He_n(x) * np.exp(-x ** 2 / 2)
        
        return const * function / np.sqrt(s)

    def fourier_period(self, s):
        """Equivalent Fourier period of derivative of Gaussian"""
        return 2 * np.pi * s / (self.m + 0.5) ** .5

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError()
        
class Poisson(object):
    def __init__(self, m=2):
        """Initialise a Poisson wavelet of order `m`."""
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return a Derivative of Gaussian wavelet,
        When m = 2, this is also known as the "Mexican hat", "Marr"
        or "Ricker" wavelet.
        It models the function::
            ``A d^m/dx^m exp(-x^2 / 2)``,
        where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5``
        and   ``x = t / s``.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        Parameters
        ----------
        t : float
            Time. If `s` is not specified, this can be used as the
            non-dimensional time t/s.
        s : scalar
            Width parameter of the wavelet.
        Returns
        -------
        out : float
            Value of the DOG wavelet at the given time
        Notes
        -----
        The derivative of the Gaussian has a polynomial representation:
        from http://en.wikipedia.org/wiki/Gaussian_function:
        "Mathematically, the derivatives of the Gaussian function can be
        represented using Hermite functions. The n-th derivative of the
        Gaussian is the Gaussian function itself multiplied by the n-th
        Hermite polynomial, up to scale."
        http://en.wikipedia.org/wiki/Hermite_polynomial
        Here, we want the 'probabilists' Hermite polynomial (He_n),
        which is computed by scipy.special.hermitenorm
        """
        m = self.m
        x = (t + (m - np.sqrt(m))) / s

        # compute the Poisson wavelet
        Un_n = lambda x: 0.5 * (np.sign(x) + 1);
        
        norm = (4**(-m) * m * scipy.special.gamma(2*m-1))/(math.factorial(m)**2)
        const = norm / math.factorial(m)
        function = - Un_n(x) * (x-m) * (x ** (m-1)) * np.exp(-x);
        
        return const * function # / np.sqrt(s)

    def fourier_period(self, s):
        """Equivalent period ?"""
        return 2 * np.pi * s

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError
        
class Exponential(object):
    def __init__(self, sigma = 10):
        """Initialise a exponential wavelet."""
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return exponential like wavelet
        """
        x = t / s

        # compute the fake exponential wavelet
        Un_n = lambda x: 0.5 * (np.sign(x) + 1);
        
        const = 1 
        function = Un_n(x) * (np.exp(-x) - 2/(self.sigma * np.sqrt(2*np.pi)) * np.exp(-x**2 / (2*self.sigma**2)));
                
        return const * function;
        
    def fourier_period(self, s):
        """Equivalent period ?"""
        return 2 * np.pi * s

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError()

class APTFunction(object):
    def __init__(self, prop = 1.0/0.05, sigma = 10):
        """Initialise a exponential wavelet."""
        self.prop = prop
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return exponential like wavelet
        """
        x = t / s
        b = self.prop

        # compute the fake exponential wavelet
        Un_n = lambda x: 0.5 * (np.sign(x) + 1)
        gauss_const = (8 * (3 + b * (3 + b)) * np.exp(-b) / b**4)
        gauss_func = 2/(self.sigma * np.sqrt(2*np.pi)) * np.exp(-x**2 / (2*self.sigma**2))
        
        m = 2
        He_n = scipy.special.hermitenorm(m)
        gamma = scipy.special.gamma

        ricker_const = (-1) ** (m + 1) / gamma(m + 0.5) ** .5
        ricker_func = He_n(x) * np.exp(-x ** 2 / 2)
        
        const = 1/np.sqrt(2.2851245589604852*10**(-20))
        function = Un_n(x) * (np.exp(-b*np.sqrt(1+np.sqrt(np.abs(x)))) - gauss_const * gauss_func)
                
        return const * function
        
    def fourier_period(self, s):
        """Equivalent period ?"""
        return 2 * np.pi * s

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError()
        
class APTFunction2(object):
    def __init__(self, prop = 1.0/0.05, sigma = 10):
        """Initialise a exponential wavelet."""
        self.prop = prop
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return exponential like wavelet
        """
        x = t / s
        b = self.prop

        # compute the fake exponential wavelet
        Un_n = lambda x: 0.5 * (np.sign(x) + 1)
        gauss_const = (8 * (3 + b * (3 + b)) * np.exp(-b) / b**4)
        gauss_func = 2/(self.sigma * np.sqrt(2*np.pi)) * np.exp(-x**2 / (2*self.sigma**2))
        
        m = 2
        He_n = scipy.special.hermitenorm(m)
        gamma = scipy.special.gamma

        ricker_const = (-1) ** (m + 1) / gamma(m + 0.5) ** .5
        ricker_func = He_n(x) * np.exp(-(x ** 2 / 2) * 1000000)
        
        const = 1/np.sqrt(2.2851245589604852*10**(-20))
        function = Un_n(x) * (np.exp(-b*np.sqrt(1+np.sqrt(np.abs(x)))) - gauss_const * gauss_func)
                
        return const * function + 30 * Un_n(x) * ricker_const * ricker_func;
        
    def fourier_period(self, s):
        """Equivalent period ?"""
        return 2 * np.pi * s

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError()
        
class APTFunctionLog(object):
    def __init__(self, sigma = 10):
        """Initialise a exponential wavelet."""
        self.sigma = sigma

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return exponential like wavelet
        """
        x = t / s

        # compute the fake exponential wavelet
        Un_n = lambda x: 0.5 * (np.sign(x) + 1);
        
        const = 1 
        function = Un_n(x) * (-1/0.05*np.sqrt(1+np.sqrt(np.abs(x))));
                
        return const * function;
        
    def fourier_period(self, s):
        """Equivalent period ?"""
        return 2 * np.pi * s

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        raise NotImplementedError()

    def coi(self, s):
        raise NotImplementedError()
