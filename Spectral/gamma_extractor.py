# -*- coding: utf-8 -*-
"""
Created on Tue May  5 17:11:19 2020

@author: TOMAS (UDEA-FAU)
"""
import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter

### UTILITY FUNCTIONS ###
def erb_space(low_freq=50, high_freq=8000, n=64):
    ear_q = 9.26449
    min_bw = 24.7
    cf_array = -(ear_q * min_bw) + np.exp(np.linspace(1,n,n) * (-np.log(high_freq + ear_q * min_bw) + np.log(low_freq + ear_q * min_bw)) / n) \
                * (high_freq + ear_q * min_bw)
    return cf_array

def powerspec(X,n_padded):    
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.abs(Y) ** 2, n_padded

### GAMMATONE IMPULSE RESPONSE ###

def gammatone_impulse_response(samplerate_hz, length_in_samples, center_freq_hz,p):
    # Generate a Glasberg&Moore parametrized gammatone filter
    erb = 24.7 + (center_freq_hz/9.26449) # equivalent rectangular bandwidth.
    #Center frequencies
    
    an = (np.pi * np.math.factorial(2*p-2) * np.power(2, float(-(2*p-2))) )/ np.square(np.math.factorial(p-1))
    b = erb/an # bandwidth parameter
    
    a = 1 # amplitude. This is varied later by the normalization process.
    t = np.linspace(1./samplerate_hz, length_in_samples/samplerate_hz, length_in_samples)
    gammatone_ir = a * np.power(t, p-1)*np.exp(-2*np.pi*b*t) * np.cos(2*np.pi*center_freq_hz*t)
    return gammatone_ir

### MP-GTF CONSTRUCTION ###

def generate_filterbank(fs,fmax, L, N,p=4):
    """
    L: Size of the signal measured in samples
    N: Number of filters
    p: Order of the Gammatone impulse response
    """
    #Center frequencies
    if fs==8000:
        fmax = 4000
    center_freqs = erb_space(50,fmax, N)
    center_freqs = np.flip(center_freqs)
    n_center_freqs = len(center_freqs)
    
    # Initialize variables
    filterbank = np.zeros((N, L))
    
    # Generate all filters for each center frequencies
    for i in range(n_center_freqs):
        filterbank[i, :] = gammatone_impulse_response(fs, L, center_freqs[i],p)
    return filterbank

def gfcc(cochleagram,numcep=13):
    feat = dct(cochleagram, type=2, axis=1, norm='ortho')[:,:numcep]
#    feat-= (np.mean(feat, axis=0) + 1e-8)#Cepstral mean substration
    return feat


def cochleagram(sig_spec,filterbank,nfft):
    """
    sig_spec: It's the STFT of the speech signal
    """
    filterbank,_ = powerspec(filterbank, nfft)#|FFT|
    filterbank /= np.max(filterbank, axis=-1)[:, None]#Normalize filters
    cochlea_spec = np.dot(sig_spec,filterbank.T)
    cochlea_spec = np.where(cochlea_spec == 0.0, np.finfo(float).eps, cochlea_spec)
#    cochlea_spec= np.log(cochlea_spec)-np.mean(np.log(cochlea_spec),axis=0)
    cochlea_spec= np.log(cochlea_spec)
    return cochlea_spec,filterbank