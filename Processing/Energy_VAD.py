# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:55:29 2021

@author: TOMAS
"""

import numpy as np
import scipy as sp
from scipy.signal import gaussian

def eVAD(sig,fs,win=0.015,step=0.01):
    """
    Energy-based Voice Activity Detection
    """    
    #Normalize signal
    sig = sig-np.mean(sig)
    sig /=np.max(np.abs(sig))
    
    lsig = len(sig)
    #Add silence to the beginning and end in case the user is an idiot or myself
    
    #Set min threshold base on the energy of the signal
    e = []
    frames = extract_windows(sig,int(win*fs),int(step*fs))
    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))
    e = np.asarray(e)
    idx_min = np.where(e==np.min(e))
    thr = np.min(frames[idx_min])
    
    ext_sil = int(fs)
    esil = int((ext_sil/2)/fs/step)
    new_sig = np.random.randn(lsig+ext_sil)*thr
    new_sig[int(ext_sil/2):lsig+int(ext_sil/2)] = sig
    sig = new_sig

    e = []#energy in dB
    frames = extract_windows(sig,int(win*fs),int(step*fs))
    frames*=np.hanning(int(win*fs))
    for seg in frames:
        e.append(10*np.log10(np.sum(np.absolute(seg)**2)/len(seg)))
    
    e = np.asarray(e)
    e = e-np.mean(e)
    #Smooth energy contour to remove small energy variations
    gauslen = int(fs*0.01)
    window = gaussian(gauslen, std=int(gauslen*0.05))
    #Convolve signal with Gaussian window for smmothing
    smooth_env = e.copy()
    smooth_env = sp.convolve(e,window)
    smooth_env = smooth_env/np.max(smooth_env)
    ini = int(gauslen/2)
    fin = len(smooth_env)-ini
    e = smooth_env[ini:fin]
    e = e/np.max(np.abs(e))
    e = e[esil:int(lsig/fs/step)+esil]
    
    thr = np.median(e[e<0])
    
    cont_sil = np.zeros(lsig)
    cont_vad = np.zeros(lsig)
    itime = 0
    etime = int(win*fs)
    for i in range(len(e)):
        if e[i]<=thr:
            cont_sil[itime:etime] = 1
        else:
            cont_vad[itime:etime] = 1
            
        itime = i*int(step*fs)
        etime = itime+int(win*fs)
    
    sig = sig[int(ext_sil/2):lsig+int(ext_sil/2)]#Remove silence added at the begining
    if np.sum(cont_sil)!=0:
        #Pauses
        dur_sil,seg_sil,time_sil = get_segments(sig,fs,cont_sil)
        #Voice
        dur_vad,seg_vad,time_vad = get_segments(sig,fs,cont_vad)
    else:
        dur_sil = [0]
        seg_sil = [0]
        dur_vad = [0]
        seg_vad= [0]
    
    X_vad = {'Pause_labels':cont_sil,
             'Pause_duration':dur_sil,
             'Pause_segments':seg_sil,
             'Pause_times':time_sil,
             'Speech_labels':cont_vad,
             'Speech_duration':dur_vad,
             'Speech_segments':seg_vad,
             'Speech_times':time_vad}
    return X_vad


def get_segments(sig,fs,segments):
        segments[0] = 0
        segments[-1:] = 0
        yp = segments.copy()
        ydf = np.diff(yp)
        lim_end = np.where(ydf==-1)[0]+1
        lim_ini = np.where(ydf==1)[0]+1
        #Silence segments
        seg_dur = []#Segment durations
        seg_list = []#Segment list
        seg_time = []#Time stamps
        for idx in range(len(lim_ini)):
            #------------------------------------
            tini = lim_ini[idx]/fs
            tend = lim_end[idx]/fs
            seg_dur.append(np.abs(tend-tini))
            seg_list.append(sig[lim_ini[idx]:lim_end[idx]])
            seg_time.append([tini,tend])
            
        seg_dur = np.asarray(seg_dur)
        seg_time = np.vstack(seg_time)
        return seg_dur,seg_list,seg_time
    
def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert(signal.ndim == 1)
    
#    # subtract DC (also converting to floating point)
#    signal = signal - signal.mean()
    
    n_frames = int((len(signal) - size) / step)
    
    # extract frames
    windows = [signal[i * step : i * step + size] 
               for i in range(n_frames)]
    
    # stack (each row is a window)
    return np.vstack(windows)

    
    