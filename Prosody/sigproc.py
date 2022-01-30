"""
This file contains different useful funtions for signal processing
"""
import os
import numpy as np
import math
from scipy.io.wavfile import read #Leer y guardar audios
import scipy as sp
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert, gaussian
from scipy.signal import firwin,lfilter

def check_audio(audio_path):
    """
    Obtener frecuencia de muestreo y numero de canales
    Entrada
        :param audio_path: Carpeta que contiene los audios
    Salida
        :returns Frecuencia de muestreo promedio en la carpeta de audios
    """
    file_list = os.listdir(audio_path)
    list_fs = []
    for audio_name in file_list:
        fs,sig = read(audio_path+'/'+audio_name)
        list_fs.append(fs)
        channels = len(sig.shape)
        print('Audio: '+audio_name+' Fs: '+str(fs)+' Canales: '+str(channels))
    print('Fs Maximo: '+str(np.max(list_fs)))
    print('Fs Minimo: '+str(np.min(list_fs)))
    print('Fs promedio: '+str(np.mean(list_fs)))
    return np.mean(list_fs)

#=========================================================
def static_feats(featmat):
    """Compute static features
    :param featmat: Feature matrix
    :returns: statmat: Static feature matrix
    """
    mu = np.mean(featmat,0)
    st = np.std(featmat,0)
    ku = kurtosis(featmat,0)
    sk = skew(featmat,0)    
    statmat = np.hstack([mu,st,ku,sk])    
    return statmat.reshape(1,-1)

#============================================================
def min_max(x,a=1,b=0):
    """
    x = Array or matrix to normalize
    a = upper limit
    b = lower limit
    """
    if len(x.shape)==1:
        x = a+((x-np.min(x))*(b-a))/(np.max(x)-np.min(x))
    else:
        x = a+((x-np.min(x,0))*(b-a))/(np.max(x,0)-np.min(x,0))
    return x

#==========================================================
def vad(sig,fs,win,step):
    """
    The energy is computed at frame level
    
    output: sil: All detected silence segments
            sil_seg: Duration of silence segments IN BETWEEN speech, i.e., the silence
                     segments at the start/end of the audio are removed
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
#    print('e[dB]',np.min(e))
    idx_min = np.where(e==np.min(e))
#    idx_min = np.where(e<=-50)[0]#Threshold in -50dB
    thr = np.min(frames[idx_min])
#    print(np.min(e))
    
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
    
    speech = np.zeros(lsig)
    segs_vad=[]
#    index=[]
    itime = 0
    etime = int(win*fs)
    sig=sig[int(ext_sil/2):lsig+int(ext_sil/2)]
    for i in range(len(e)):
#        if e[i]<=thr:
        if e[i]>np.median(e[e<0]):
            speech[itime:etime] = 1

        itime = i*int(step*fs)
        etime = itime+int(win*fs)
#    ivad = np.where(e>np.mean(e[e<0]))[0]
    speech[0] = 0
    speech[-1:] = 0
    
    yp = speech.copy()
    ydf = np.diff(yp)
    lim_end = np.where(ydf==-1)[0]+1
    lim_ini = np.where(ydf==1)[0]+1
    times_stamps=[]
    sil_seg = []
    for idx in range(len(lim_ini)):
        #------------------------------------
        tini = lim_ini[idx]/fs
        tend = lim_end[idx]/fs
        sil_seg.append(np.abs(tend-tini))
        segs_vad.append(sig[lim_ini[idx]:lim_end[idx]])
        times_stamps.append([lim_ini[idx]/fs,lim_end[idx]/fs])
    return speech,segs_vad, np.vstack(times_stamps)
#=========================================================
def add_noise(sig,target_snr_db=10):
    print('!'*50)
    print('Adding Gaussian noise of',target_snr_db,'dB to the signal')
    print('!'*50)
    #Remove DC level and re-scale between 1 and -1
    sig = norm_sig(sig)
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
#=========================================================
def norm_sig(sig):
    """Remove DC level and scale signal between -1 and 1.

    :param sig: Signal to normalize
    :returns: Normalized signal
    """
    #Eliminar nivel DC
    normsig = sig-np.mean(sig)
    #Escalar valores de amplitud entre -1 y 1
    normsig = normsig/float(np.max(np.absolute(normsig)))
    return normsig

def sig_contour(cont_list,sig,fs,win=0.04,solp=0.01):
     """Get contorus to plot along the speech signal
     :param cont_list: List with pitch values, energy,.....
     :param sig: Speech signal
     :param fs: sampling frequency
     :param win: win length in miliseconds
     :param solp: step size in miliseconds
     :returns: contour
     """
     cont = np.zeros(len(sig))
     siz = int(fs*solp)
     ini = 0
     end = siz+ini
     for i in cont_list:
          cont[ini:end] = i
          ini = end
          end = end+siz
#     g = float(max(np.absolute(min(sig)),max(sig)))#Valor maximo para escalar energía
#     cont = (g*cont)/float(max(np.absolute(min(cont)),max(cont)))
     return cont

def framesig(sig,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len: 
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0*slen - frame_len)/frame_step))
        
    padlen = int((numframes-1)*frame_step + frame_len)
    
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig,zeros))
    
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len),(numframes,1))
    return frames*win
    
    
def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:np.ones((1,x))):
    """Does overlap-add procedure to undo the action of framesig. 

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.    
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.    
    :returns: a 1-D signal.
    """
    frame_len = round(frame_len)
    frame_step = round(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = np.tile(np.arange(0,frame_len),(numframes,1)) + np.tile(np.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = np.array(indices,dtype=np.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = np.zeros((1,padlen))
    window_correction = np.zeros((1,padlen))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """    
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)
          
def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """    
    return 1.0/NFFT * np.square(magspec(frames,NFFT))
    
def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT. 

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded. 
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """    
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps
    
def preemphasis(signal,coeff=0.95):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def highpass_fir(sigR,fs,fc,nfil):
    #sigR: Sennal a filtrar
    #fs: Frecuencia de muestreo de la sennal a filtrar
    #fc: Frecuencia de corte.
    #nfil: Orden del filtro
    largo = nfil+1 #  orden del filtro
    fcN = float(fc)/(float(fs)*0.5) # Frecuencia de corte normalizada
    #Filtro pasa bajas
    h = firwin(largo, fcN)
    #Inversion espectral para obtener pasa altas    
    h = -h
    h[int(largo/2)] = h[int(largo/2)] + 1
    #Aplicar transformada
    sigF = lfilter(h, 1,sigR)
    return sigF

#*****************************************************************************
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
#*****************************************************************************
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
#*****************************************************************************
def powerspec(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
    return np.abs(Y) ** 2, n_padded
#*****************************************************************************
def powerspec2D(X, rate, win_duration, n_padded_min=0):
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    
#    Y_real = np.abs(np.diff(Y.real,axis=1))
#    Y_img = np.abs(np.diff(Y.imag,axis=1))
#    Y_real = np.hstack([Y_real,Y_real[:,-1:]])
#    Y_img = np.hstack([Y_img,Y_img[:,-1:]])
    
    c = np.finfo(np.float).eps
    Y_real = np.abs(Y.real)+c
    Y_img = np.abs(Y.imag)+c
    mag = np.sqrt((Y_img)**2+(Y_real)**2)
    phase = np.arctan(Y_img.copy(),Y_real.copy())
    return mag,phase
#*****************************************************************************
def powerspec3D(X, rate, win_duration, n_padded_min=0):
    """
    Output: the power, magnitude, and phase spectrums of X
    """
    win_size = int(rate * win_duration)
    
    # apply hanning window
    X *= np.hanning(win_size)
    
    # zero padding to next power of 2
    if n_padded_min==0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(win_size) / np.log(2))))
    else:
        n_padded = n_padded_min
    # Fourier transform
#    Y = np.fft.rfft(X, n=n_padded)
    Y = np.fft.fft(X, n=n_padded)
#    Y = np.absolute(Y)
    
    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]
    power = np.abs(Y) ** 2
    Y_real = np.abs(Y.real)
    Y_img = np.abs(Y.imag)
    mag = np.sqrt((Y_img)**2+(Y_real)**2)
    phase = np.arctan(Y_img.copy(),Y_real.copy())
    return power,mag,phase
#********************************************************
def read_file(file_name):
    """
    Converts the text in a txt, txtgrid,... into a python list
    """
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n','')
    return lines
#****************************************************************************
def get_file(path,cond):
    """
    path: Folder containing the file name
    cond: Name, code, or number contained in the filename to be found:
          If the file name is 0001_ADCFGT.wav, cond could be 0001 or ADCFGT.
    """
    list_files = os.listdir(path)
    filesl = []
    for f in list_files:
        if f.upper().find(cond.upper())!=-1:
            filesl.append(f)
    if len(filesl)==1:
        f = filesl[0]
    elif len(filesl)>1:
        f = filesl
    else:
        f = ''#If no file is found, then return blank
    return f
#########################################
def hilb_tr(signal,fs,smooth=True,glen = 0.01):
    """
    Apply hilbert transform over the signal to get
    the envelop and time fine structure
    
    If smooth true, then the amplitude envelope is smoothed with a gaussian window
    """
    #Hilbert Transform
    analytic_signal = hilbert(signal)
    #Amplitude Envelope
    amplitude_envelope = np.abs(analytic_signal)
    
    #Temporal Fine Structure
    tfs = analytic_signal.imag/amplitude_envelope
    
    #Convolve amplitude evelope with Gaussian window
    if smooth==True:
        #Gaussian Window
        gauslen = int(fs*glen)
        window = gaussian(gauslen, std=int(gauslen*0.05))
        #Convolve signal for smmothing
        smooth_env = amplitude_envelope.copy()
        smooth_env = sp.convolve(amplitude_envelope,window)
        smooth_env = smooth_env/np.max(smooth_env)
        ini = int(gauslen/2)
        fin = len(smooth_env)-ini
        amplitude_envelope = smooth_env[ini:fin]
    return amplitude_envelope,tfs

def ACF_spec(sig,fs,filterbank,nfft=1024,win_time=0.04,step_time=0.01):
    """
    sig: Audio signal
    filterbank: As computed for the MFCC or Gammatone
    """
    frames = extract_windows(sig, int(win_time*fs), int(step_time*fs))
    frames *= np.hanning(int(win_time*fs))
    #Autocorrelation
    Rsig = np.zeros((frames.shape[0],int(nfft/2)+1))
    i = 0
    for fr in frames:
    
        ra = np.correlate(fr, fr, mode='full')
        ra = ra[int(ra.size/2):]#Only half
        ra/=np.max(ra)
    
        if len(ra)<int(nfft/2)+1:
            Rsig[i,0:len(ra)] = ra
        else:
            Rsig[i,:] = ra[0:int(nfft/2)+1]
    
        i+=1
    
    Rsig = np.vstack(Rsig)
    Rsig  = np.dot(Rsig ,filterbank.T)
    return Rsig