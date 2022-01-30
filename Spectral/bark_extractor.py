# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:27:48 2020

@author: TOMAS
"""
import numpy as np
import math 

def barke(y,Fs,nB=17,nfft=512):
    """
    y: log-STFT of the signal [framesXbins]
    Fs: sampling frequency
    nfft: number of points for the Fourier transform
    nB: number of bands for energy computation (default is 17)
    """
    f = (Fs/2)*(np.linspace(0,1,nfft/2+1))
    barkScale = bark_bands(f)
    
    barkIndices = []
    for i in range (0,len(barkScale)):
        barkIndices.append(int(barkScale[i]))
    barkIndices = np.asarray(barkIndices)
    
    if len(y.shape)>1:#In case the signal is framed 
        BarkE = []
        for idx_y in y:       
            BarkE.append(bark_idx(barkIndices,nB,idx_y))
    else:
        BarkE = bark_idx(barkIndices,nB,y)
    return np.vstack(BarkE)

def bark_bands(f):
	x=(f*0.00076)
	x2=(f/7500)**2
	b=[]
	for i in range (0,len(f)):
		b.append(13*( math.atan(x[i]) )+3.5*( math.atan(x2[i]))) #Bark scale values
	return (b) 

def bark_idx(barkIndices,nB,sig):
    barkEnergy=[]    
#    eps = 1e-30 
    for i in range (nB):
        brk = np.nonzero(barkIndices==i)
        brk = np.asarray(brk)[0]
        sizeb=len(brk)
        if (sizeb>0):
            barkEnergy.append(sum(np.abs(sig[brk]))/sizeb)
        else:
            barkEnergy.append(0)        
    e = np.asarray(barkEnergy)#+eps
    return e