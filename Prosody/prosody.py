
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017, Modified Nov 15 2019.

@authors: T. Arias-Vergara

Compute prosody features based on pitch, loudness, duration, ratios, rhythm, and perturbations (apq/ppq)

OUTPUT OF THE FUNCTION "prosody_features":

"""


import os
path_base = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import warnings
import sigproc as sg
import scipy as sp
#from scipy.stats import kurtosis, skew
from scipy.signal import gaussian
from scipy.io.wavfile import write 
import praat.praat_functions as praatF 
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

def prosody_features(sig,fs,f0=np.asarray([0]),winTime=0.04,stepTime=0.01):
    if (np.sum(f0)==0)&(len(f0)==1):
        f0 = f0_contour_pr(sig,fs,winTime,stepTime)#F0
    #VAD
    out_VAD = eVAD(sig,fs)
    
    #Compute f0 features
    feats_f0 = f0_features(sig,fs,f0,winTime,stepTime)
    
    #Compute voiced features
    feats_voiced,vcont = voiced_features(sig,fs,f0,stepTime)
    
    #Compute VAD features (duration+energy content)
    feats_VAD = VAD_features(sig,fs,out_VAD,winTime,stepTime)
    
    #Compute unvoiced features
    feats_unvoiced = unvoiced_features(sig,fs,vcont,out_VAD['Pause_labels'])
    
    
    X = [feats_f0,feats_voiced,feats_unvoiced,feats_VAD]
    
    #Create new dictionary with all features
    X_pr = {}
    for k in X:
        for f in list(k.keys()):
            X_pr[f] = k[f]
    return X_pr

def prosody_features_dynamic(sig,fs,f0=np.asarray([0]),winTime=0.04,stepTime=0.01):
    if len(f0)==0:
        f0 = f0_contour_pr(sig,fs,winTime,stepTime)#F0
    #---------------------------------------
    f0coef,voiced,_ = voiced_unvoiced(sig,fs,f0,stepTime)
#    f0coef = np.vstack(f0coef)
    #Voiced features
    lvoiced = []
    for v in voiced:
        lvoiced.append(len(v)/fs)#Length of voiced segment
    lvoiced = np.vstack(lvoiced)
    #.........................................................
    X = np.hstack([lvoiced,f0coef])
    return X
#==========================================================================
def Hz2Semitone(F):
    ST=39.87*np.log(F/50)
    return ST
#==========================================================================
def f0_contour_pr(sig,fs,sizeframe=0.04,step=0.01,maxf0=500, post=False):
    """
    This function is used to extract the F0 contour using praat 
    """    
    sig = sig-np.mean(sig)
    sig = sig/np.max(np.abs(sig))
    
    
    temp_aud =  (sig*2**15).astype(np.int16)
    temp_path = path_base+'\\temp_sig.wav'#Creates temporal wav file
    write(temp_path,int(fs),temp_aud)
    temp_filename_f0=path_base+'/praat/tempF0.txt'
    np.savetxt(temp_filename_f0,np.zeros((3,3))) 
    temp_filename_vuv=path_base+'/praat/tempvuv.txt'
    np.savetxt(temp_filename_vuv,np.zeros((3,3)))  
    
    minf0 = int(3/sizeframe)
    praatF.praat_vuv(temp_path, temp_filename_f0, temp_filename_vuv, 
                              time_stepF0=step, minf0=minf0, maxf0=maxf0)
    #Tomas: I modified this function. The size of the frame (in seconds) and sampling frequency are 
    #now input arguments. This was neccesary to compute the number of frames correctly.
    f0,_ = praatF.decodeF0(temp_filename_f0,len(sig),float(fs),sizeframe,step)
    if np.sum(f0)==0:
        print('PITCH WAS NOT DETECTED')
    os.remove(temp_filename_f0)    
    os.remove(temp_filename_vuv) 
    os.remove(temp_path)
    
    #Post-processing of F0 to avoid outliers. Is very simple
    if post==True:
        print('F0 post-processing Activated')
        uf0 = np.mean(f0[f0>0])
        sf0 = np.std(f0[f0>0])
        f0[f0>(uf0+(2.5*sf0))] = 0
        f0[f0<(uf0-(2.5*sf0))] = 0
    return f0
#==========================================================================
def voiced_unvoiced(sig,fs,f0,stepTime):
    """
    Voiced unvoiced segmentation
    sig: Speech signal
    fs: Sampling frequency
    f0: Pitch contour
    stepTime: Step size (in seconds) used to computed the f0 contour.
    """
    yp = f0.copy()
    yp[yp!=0] = 1
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    #Voiced segments
    v_segm = []
    f0_feats = []#Dynamic f0-based features
    #Unvoiced
    uv_segm = []
    for idx in range(len(lim_ini)):
        #------------------------------------
        #Voiced segments
        tini = int(lim_ini[idx]*stepTime*fs)
        tend = int(lim_end[idx]*stepTime*fs)
        if int(tend-tini)>int(0.04*fs):
#            print(tini,tend)
            v_segm.append(sig[tini:tend])
            x = np.arange(0,len(f0[lim_ini[idx]:lim_end[idx]]))
            #F0 based features
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                f0c = np.polyfit(x,f0[lim_ini[idx]:lim_end[idx]],5)
#            f0c = f0c.reshape(1,-1)#Dynamic reprsentation of f0.
            p = np.poly1d(f0c)
            f0_mse = mse(f0[lim_ini[idx]:lim_end[idx]],p(x))         
#            plt.plot(p(x),'k',label='Fitted')
#            plt.plot(f0[lim_ini[idx]:lim_end[idx]],'r',label='Real')
#            plt.legend()
            if len(sig[tini:tend])>int(3*0.04*fs):
                frames = sg.extract_windows(sig[tini:tend],int(0.04*fs),int(0.01*fs))
                jitter = ppq(f0[lim_ini[idx]:lim_end[idx]],3)
                ak = np.max(frames,axis=1)
                shimmer = apq(ak,3)
            else:
                jitter = 0
                shimmer = 0
            f0temp = np.hstack([jitter,shimmer,len(sig[tini:tend])/fs,f0_mse,f0c])
            f0_feats.append(f0temp)
            #--------------------------------
        #------------------------------------
        #Unvoiced segments
        tini = int(lim_end[idx]*stepTime*fs)
        if (idx+1)<(len(lim_ini)-1):
            tend = int(lim_ini[idx+1]*stepTime*fs)
            if int(tend-tini)<int(0.27*fs):
                uv_segm.append(sig[tini:tend])
    #--------------------------------------------------------------------
    f0_feats = np.vstack(f0_feats)
    return f0_feats,v_segm,uv_segm
#==========================================================================
def voiced_seg(sig,fs,f0,stepTime):
    """
    Voiced segments
    sig: Speech signal
    fs: Sampling frequency
    f0: Pitch contour
    stepTime: Step size (in seconds) used to computed the f0 contour.
    """
    yp = f0.copy()
    yp[yp!=0] = 1
    #In case the starting point is F0 and not 0
    if yp[0] == 1:
        np.insert(yp, 0, 1)
    if yp[-1:] == 1:
        np.insert(yp, 0, len(yp)-1)
    #---------------------
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    #Voiced segments
    v_segm = []
    tm = []
    vcont = np.zeros(len(sig))
    for idx in range(len(lim_ini)):
        #------------------------------------
        
        #Voiced segments
        tini = int(lim_ini[idx]*stepTime*fs)
        tend = int(lim_end[idx]*stepTime*fs)
        if int(tend-tini)>int(0.04*fs):
#            print(tini,tend)
            vcont[tini:tend] = 1
            v_segm.append(sig[tini:tend])
            tm.append(np.hstack([lim_ini[idx]*stepTime,lim_end[idx]*stepTime]))
            
    vseg = {'Voiced_segments':v_segm,
            'Voiced_times':tm,
            'Voiced_labels':vcont}
    return vseg
#----------------------------------------------------------------------------
def unvoiced_seg(sig,fs,vseg,sil):
    uvcont = sil+vseg+1
    uvcont[uvcont>1] = 0
    uvcont[0] = 0
    uvcont[-1:] = 0
    yp = uvcont.copy()
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    #Voiced segments
    uv_seg = []
    uv_dur = []
    uv_tm = []
    for idx in range(len(lim_ini)):
        #------------------------------------
        try:
            tini = lim_ini[idx]/fs
            tend = lim_end[idx]/fs
    #        uv_dur.append(tend-tini)
            uv_seg.append(sig[lim_ini[idx]:lim_end[idx]])
            uv_tm.append([tini,tend])
        except:
                print('Unvoiced segment not included')
    uv_dur = np.asarray(uv_dur)
    return uv_seg,uv_tm,uvcont
#----------------------------------------------------------------------------
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
    frames = sg.extract_windows(sig,int(win*fs),int(step*fs))
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
    frames = sg.extract_windows(sig,int(win*fs),int(step*fs))
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
#----------------------------------------------------------------------------
def decodef0_transitions(sig,fs,f0,trans,sztr=0.16,step=0.01):
    """
    F0 is the pitch contourn
    trans = onset or offset
    sztr: Size of the transition. Default is 160 ms:80 ms voiced; 80 ms unvoiced
    step: The step used to compute the f0 contourn of the signal
    """
    if trans.lower()=='onset':
        trflag=1
    elif trans.lower()=='offset':
        trflag=-1
    else:
        return print('Options in trans: onset or offset')
    modf0 = f0.copy()
    modf0[modf0>0] = 1
    #f0 will be found were modf0!=0
    f0diff = np.diff(modf0)
    #transitions will be found where f0diff=trflag
    idx_tr = np.where(f0diff==trflag)[0]
    idx_tr = idx_tr+1#Compensate 1 for the np.diff operation
    tm = [] #Time stamps
    seg_tr = [] #Onset or Offset segment
    winl = int(sztr*fs/2)#Size of the transition in samples
    for iseg in idx_tr: 
        t1 = int(iseg*step*fs-winl)
        t2 = int(iseg*step*fs+winl)
        seg = sig[t1:t2]
        if len(seg)>=int(fs*sztr):
            seg_tr.append(seg)
            tm.append([t1/fs,t2/fs])
    return seg_tr,tm

def f0_features(sig,fs,f0=np.asarray([0]),winTime=0.04,stepTime=0.01):
    if (np.sum(f0)==0)&(len(f0)==1):
        f0 = f0_contour_pr(sig,fs,winTime,stepTime)#F0
    #---------------------------------------
    #F0 FEATURES
    uf0 = np.mean(f0[f0>0])
    sf0 = np.std(f0[f0>0])
    #F0 in semitones
#    ust = Hz2Semitone(uf0)
#    sst = Hz2Semitone(sf0)
    
#    feats_f0 = np.hstack([uf0,sf0,ust,sst])
    
    feats_f0 ={'F0_mean':uf0,
               'F0_std':sf0}
    return feats_f0

def voiced_features(sig,fs,f0,stepTime):
    """
    Voiced segment features
    """
    vsegs = voiced_seg(sig,fs,f0,stepTime)
    #Voiced features
    lvoiced = []
    for v in vsegs['Voiced_segments']:
        lvoiced.append(len(v)/fs)#Length of voiced segment
    uvoiced = np.mean(lvoiced)#Average length
    vrate = (len(vsegs['Voiced_segments'])*fs)/len(sig)#Voiced segments per second
    numv = len(vsegs['Voiced_segments'])

    #Rhythm -based
    rPVI,nPVI = get_pvi(lvoiced)
    pGPI,dGPI = get_gpi(lvoiced,len(sig)/fs) #pGPI = Voiced rate
    
#    feats_voiced = np.hstack([numv,vrate,uvoiced,rPVI,nPVI,pGPI,dGPI])
    
    feats_voiced = {'Voiced_counts':numv,
                    'Voiced_rate':vrate,
                    'Voiced_duration':uvoiced,
                    'Voiced_rPVI':rPVI,
                    'Voiced_nPVI':nPVI,
                    'Voiced_dGPI':dGPI}
    return feats_voiced,vsegs['Voiced_labels']

def unvoiced_features(sig,fs,vcont,sil_cont):
    """
    Unvoiced segment features.
    
    Requires voiced and silence/pauses segment detection.
    """
    #Unvoiced features
    uv_seg,_,_ = unvoiced_seg(sig,fs,vcont,sil_cont)
    lunvoiced = []
    for uv in uv_seg:
        lunvoiced.append(len(uv)/fs)#Length of unvoiced segment
    uunvoiced = np.mean(lunvoiced)#Average length
#    sunvoiced = np.std(lunvoiced)#variation of length
    uvrate = (len(uv_seg)*fs)/len(sig)#Unvoiced segments per second
    numuv = len(uv_seg)
    rPVI,nPVI = get_pvi(lunvoiced)
    pGPI,dGPI = get_gpi(lunvoiced,len(sig)/fs)
#    feats_unvoiced = np.hstack([numuv,uvrate,uunvoiced,rPVI,nPVI,pGPI,dGPI])
    
    feats_unvoiced = {'Unvoiced_counts':numuv,
                      'Unvoiced_rate':uvrate,
                      'Unvoiced_duration':uunvoiced,
                      'Unvoiced_rPVI':rPVI,
                      'Unvoiced_nPVI':nPVI,
                      'Unvoiced_dGPI':dGPI}
    return feats_unvoiced

def VAD_features(sig,fs,out_VAD,win_time=0.025,step_time=0.01):
    npause,rpause,dpause = duration_features(sig,fs,out_VAD['Pause_duration'],out_VAD['Pause_segments'])
    nspeech,rspeech,dspeech = duration_features(sig,fs,out_VAD['Speech_duration'],out_VAD['Speech_segments'])
    
    #Compute energy based features only for speech segments
    mSPL_vad,sSPL = VAD_energy_features(sig,fs,out_VAD['Speech_segments'],win_time,step_time)

    feats_vad ={'Pause_counts':npause,
                'Pause_rate':rpause,
                'Pause_duration':dpause,
                'Speech_counts':nspeech,
                'Speech_rate':rspeech,
                'Speech_duration':dspeech,
                'SPL_mean':mSPL_vad,
                'SPL_std':sSPL}
    return feats_vad
    

def duration_features(sig,fs,dsegment,segment):
    #Number of pauses, Duration of pauses, pauses per second
    dsegm = np.mean(dsegment)
    rsegm = (len(segment)*fs)/len(sig)
    nsegm = len(segment)
    return nsegm,rsegm,dsegm

def VAD_energy_features(sig,fs,seg_vad,win_time=0.025,step_time=0.01):
    """
    The SPL should be only computed for the speech segments
    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    seg_vad : TYPE
        DESCRIPTION.
    win_time : TYPE, optional
        DESCRIPTION. The default is 0.025.
    step_time : TYPE, optional
        DESCRIPTION. The default is 0.005.

    Returns
    -------
    mSPL_vad : TYPE
        DESCRIPTION.
    sSPL : TYPE
        DESCRIPTION.

    """
    SPL = sound_pressure_level(sig,fs,win_time,step_time)
    SPL_vad = []
    for ivad in seg_vad:
        SPL = sound_pressure_level(ivad,fs,win_time,step_time)
        SPL_vad.append(np.mean(SPL))
    mSPL_vad = np.mean(SPL_vad)
    sSPL = np.std(SPL_vad)
    return mSPL_vad,sSPL

def sound_pressure_level(sig,fs,win_time=0.025,step_time=0.01):
    """
    Sound Pressure Level as in:
        Švec JG, Granqvist S. Tutorial and Guidelines on Measurement of Sound 
        Pressure Level in Voice and Speech. Journal of Speech, Language, and Hearing Research. 
        2018 Mar 15;61(3):441-461. doi: 10.1044/2017_JSLHR-S-17-0095. PMID: 29450495.
        
    SPL = 20*log10(p/p0)
    
    20xlog refers to a root-power quantity e.g., volts, sound pressure, current...
    
    Intensity in dBs:
        ene = 10*log10(sum(x^2)/N)
    
    10xlog refers to a power quantity, i.e. quantities directly proportional to power
    
    x: speech signal
    N: lenght of x
    p = RMS value of x
    p0 = 20uPA = 0.00002 Hearing threshold
    """
    #Set a threshold based on the energy of the signal
    if len(sig)>3*int(win_time*fs):
        frames = sg.extract_windows(sig,int(win_time*fs),int(step_time*fs))
    else:
        frames = list([sig])
    SPL = []#Sound Pressure Level
    p0 = 2*(10**-5)#Hearing threshold at SLP 0dB
    for x in frames:
        #Sound Pressure Level (dBs)
        p = np.sqrt(np.sum((x)**2)/len(x))
        Lp = 20*np.log10(p/p0)
        SPL.append(Lp)
    SPL = np.asarray(SPL)
    return SPL

def ppq(f0,pq=2):
    """
    Teixeira, J. P., & Gonçalves, A. (2016). Algorithm for jitter and shimmer 
    measurement in pathologic voices. Procedia Computer Science, 100, 271-279.
    
    f0: Fundamental frequency contour
    pq: Number of points to be considered
        pq = 2 : Jitter
        pq = 3 : Relative Average Perturbation
        pq = 5 : PPQ computed every 5 points of f0        
    """
    #Non zero f0
    f0 = f0[f0>0]
    
    N = len(f0)

    ppq = []
    
    start = int(np.floor(pq/2))
    for i in range(start,N):
#        ppq.append(np.abs(f0[i]-Mp))
        if pq>1:
            neig = np.mean(f0[i-start:i+(pq-start)])
        else:
            neig = f0[i-1]
        ppq.append(np.abs(f0[i]-neig))
    ppq = np.sum(np.asarray(ppq))/(N-1)
    ppq = (100*ppq)/np.mean(f0)
    return ppq
######################################################################### 
def apq(ak,pq=2):
    """
    Teixeira, J. P., & Gonçalves, A. (2016). Algorithm for jitter and shimmer 
    measurement in pathologic voices. Procedia Computer Science, 100, 271-279.
    
    ak: Maximum amplitude of the signal
    pq: Number of points to be considered
        pq=3 : Shimmer
        pq=5 : APQ computed every 5 points        
    """       
#    ak = np.zeros(frames.shape[0])
#    for ie in range(len(ak)):
#        ak[ie] = np.max(frames[ie])
    
    
    N = len(ak)
    #Max F0
#    Ma = np.max(np.abs(ak))
    apq = []
    start = int(np.floor(pq/2))
    for i in range(start,N):
        if pq>1:
            neig = np.mean(ak[i-start:i+(pq-start)])
        else:
            neig = ak[i-1]
        apq.append(np.absolute(ak[i]-neig))
    apq = np.sum(np.asarray(apq))/(N-1)
    apq = (100*apq)/np.mean(ak)
    return apq
######################################################################### 
def get_pvi(d):
    """
    Rythm-based feature
    
    Raw and normalize Pairwise Variability Index (rPVI, nPVI) from:
    Grabe, E., & Low, E. L. (2002). Durational variability in 
    speech and the rhythm class hypothesis. Papers in laboratory 
    phonology, 7(515-546).
    
    (1) rPVI = SUM{k=1,m-1}|d_k - d_{k+1}|/(m -1)
    (2) nPVI = 100*SUM{k=1,m-1}|(d_k - d_{k+1})/(0.5*(d_k + d_{k+1}))|/(m -1)
    
    m   = number of intervals i.e., vocalic-, consonant-, voiced-,... segments
    d_k = duration of k-th interval
    
    input:
        d = list with duration of speech segments (vocalic, voiced, consonants,...)
    output:
        rPVI: Raw Pairwise Variability Index
        nPVI: Normalize Pairwise Variability Index
    """
    rPVI = 0
    nPVI = 0
    m = len(d)
    for k in range(m-1):
        rPVI += np.abs(d[k]-d[k+1])
        nPVI += np.abs((d[k]-d[k+1])/(0.5*(d[k]+d[k+1])))
    rPVI = rPVI/(m-1)
    nPVI = 100*nPVI/(m-1)
    return rPVI,nPVI

def get_gpi(d,n):
    """
    Rythm-based feature
    
    Global proportions of intervals from:
    Ramus, F., Nespor, M., & Mehler, J. (1999). 
    Correlates of linguistic rhythm in the speech 
    signal. Cognition, 73(3), 265-292.
    
    pGPI = SUM d_k/n
    
    input:
        d = list with duration of speech segments (vocalic, voiced, consonants,...)
        n = Length of the recording considering only the duration of vowels and 
            consonants [in seconds]. In the original paper, the authors do not consider 
            the silence/pause segments.
    output:
        pGPI: Global proportion of interval
        dGPI: variation of durations
    """
    pGPI = np.sum(d)/n
    dGPI = np.std(d)
    return pGPI,dGPI