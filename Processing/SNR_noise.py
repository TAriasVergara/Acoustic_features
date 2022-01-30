# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:00:42 2021

@author: TOMAS
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read #Leer y guardar audios

def add_noise(sig,target_snr_db=10):
    print('!'*50)
    print('Adding Gaussian noise of',target_snr_db,'dB to the signal')
    print('!'*50)
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.sum(np.absolute(sig)**2)/len(sig)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(sig))
    # Noise up the original signal
    y_volts = sig + noise_volts
    return y_volts

fs,sig = read('./data_example/schmetterling.wav')

sig = sig-np.mean(sig)
sig = sig/np.max(np.abs(sig))

sig2 = add_noise(sig,20)

plt.subplot(2,1,1)
plt.plot(sig)
plt.subplot(2,1,2)
plt.plot(sig2)
