#!/usr/bin/env python

import numpy as np
import grainstream as gs
from scipy.io import wavfile as wav
import scipy.signal as sig
import plotting as pl
import sys

# ---------------
# tukey
# Generates a tukey (tapered cosine) window.
# Same as the matlab definition: http://www.mathworks.co.uk/help/signal/ref/tukeywin.html
# Taken from: http://leohart.wordpress.com/2006/01/29/hello-world/
# ---------------
def tukey(window_length, alpha=0.5):
    if alpha <= 0:
        return np.ones(window_length)
    elif alpha >= 1:
        return np.hanning(window_length)
 
    x = np.linspace(0, 1, window_length)
    window = np.ones(window_length)
 
    fade_in = x < alpha/2
    window[fade_in] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[fade_in] - alpha/2)))
    fade_out = x >= (1 - alpha/2)
    window[fade_out] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[fade_out] - 1 + alpha/2))) 
 
    return window

# ---------------
# normalise
# Normalises a mono signal (aud) so the max value only touches lev.
# ---------------
def normalise(aud,lev,_print=False):
    peak = np.amax(np.abs(aud))
    if _print:
        print "Peak level is %.4f, normalising to %.2f.." % (peak,lev)
    return aud*(lev/peak)

# ---------------
# normalise_stereo
# Does the same as normalise but with a stereo input. Separate functions for optimisation.
# ---------------
def normalise_stereo(aud,lev,_print=False):
    peak = np.amax(np.abs(aud))
    if _print:
        print "Peak level is %.4f, normalising to %.2f.." % (peak,lev)
    return [aud[0]*(lev/peak),aud[1]*(lev/peak)]

# ---------------
# mixdown
# Mixes any number of stereo streams together.
# ---------------
def mixdown(streams):
    _streams = np.empty([len(streams)*2,streams[0].get_length()])
    for i in range(0,len(streams)):
        s = streams[i].get_audio()
        _streams[i] = s[0]
        _streams[len(streams)+i] = s[1]
    mixed = [np.sum(_streams[0:len(streams)],0),np.sum(_streams[len(streams):len(streams)*2],0)]
    return mixed

# ---------------
# compress
# Simple dynamic range compression, no attack or release or anything but works pretty well.
# Slides a window across a (stereo) signal and records the means of all the amplitudes in each frame.
# Then determines a threshold, which is defined as a level a certain percentage of frame amplitudes exceed.
# For example: a 'level' value of 0.2 will result in a threshold that 20% of the values exceed.
# Then the distances between frame amplitudes exceeding the threshold, from the threshold, are reduced
# according to the ratio.
# ---------------
def compress(stereo,window_size,params):
    level,ratio = params.comp_thresh,params.comp_ratio
    np.seterr(all='raise')
    stereo = np.array(stereo)
    window_means = []
    for i in range(0,stereo.shape[1],window_size):
            frame = np.abs(stereo[:,i:min(i+window_size,stereo.shape[1])])
            frame_mean = np.mean(frame)
            window_means.append(frame_mean)
        
    wm_sorted = np.sort(window_means)
    wm_knee = wm_sorted[int(len(wm_sorted) - len(wm_sorted) * level)]
    compressed = stereo
    _ratio = 1./ratio
    for i,m in enumerate(window_means):
        if m >= wm_knee:
            target_level = wm_knee + (m - wm_knee) * _ratio
            compressed[:,i*window_size:min((i+1)*window_size,stereo.shape[1])] = stereo[:,i*window_size:min((i+1)*window_size,stereo.shape[1])] * (target_level/m)
        
    if params.debug>1:
        window_means2 = []
        for i in range(0,stereo.shape[1],window_size):
            window_means2.append(np.mean(np.abs(compressed[:,i:min(i+window_size,stereo.shape[1])])))
        pl.plot_dynamic_range(window_means,window_means2)

    return compressed

# ---------------
# post_process
# Post-processing procedure. Mixes down the audio, compress, fades and normalies it
# and outputs it in a format that can be written to disk.
# ---------------
def post_process(streams,params):
    mixed = mixdown(streams)
    compressed = compress(mixed,16,params)
    normalised = normalise_stereo(compressed,params.norm_level,True)
    faded = normalised*tukey(len(normalised[0]),params.fade_size)
    # Convert to 16 bit integer format
    int_16 = [(faded[0]*32767).astype('Int16'),(faded[1]*32767).astype('Int16')]
    # Transpose the matrix (I work with it oriented the other way around)
    return np.transpose(int_16)

# ---------------
# read_audio
# Reads a wav file and takes extracts the first channel of it, if it has more than one. 
# ---------------
def read_audio(filename):
    [sample_rate,source_audio] = wav.read(filename)
    source_audio[source_audio==0] = 1 # just to prevent division by zero
    if len(source_audio.shape) == 1:
        source_channels = 1
        source_length = source_audio.shape[0]
        source_audio_f = source_audio.astype('Float16')/32767.
    else:
        [source_length,source_channels] = source_audio.shape
        source_audio_f = source_audio[:,0].astype('Float16')/32767.
        
    return [source_audio_f,sample_rate,source_length]

# ---------------
# write_audio
# Writes audio to a wav file on disk.
# ---------------
def write_audio(filename,sample_rate,audio):
    print "Writing to %s" % filename
    wav.write(filename,sample_rate,audio)
    
# ---------------
# cachedfilter / filter_cache
# Saves filter windows for each unique set of parameters, so they don't
# need to be calculated again.
# ---------------
filter_cache = []
class cachedfilter:
    def __init__(self,type,cutoff,trans,attenuation,filter):
        self.type = type
        self.cutoff = cutoff
        self.trans = trans
        self.attenuation = attenuation
        self.filter = filter

# ---------------
# filter_audio
# Applies a lowpass or highpass filter to a given audio segment.
# First checks to see if the filter is in the cache and calculates it if it isn't,
# then convolves it with the signal.
# ---------------
def filter_audio(audio,sr=44100,mode="lowpass",cutoff=4000.0,trans_width=500.0,attenuation=60.0):
    global filter_cache
    cached_filter = [fc for fc in filter_cache if fc.type == mode and fc.cutoff == cutoff and fc.trans == trans_width and fc.attenuation == attenuation]
    if len(cached_filter) == 0:
        nyquist = sr/2.
        width = trans_width/nyquist
        num_taps, beta = sig.kaiserord(attenuation,width)
        filter = sig.firwin(num_taps,cutoff/nyquist,window=('kaiser',beta))
        if mode == "highpass":
            filter = -filter
            filter[num_taps/2] += 1
        cf = cachedfilter(mode,cutoff,trans_width,attenuation,filter)
        filter_cache.append(cf)
    else:
        filter = cached_filter[0].filter
    convolved = np.convolve(audio,filter,"same")
    if len(convolved) > len(audio): # this doesn't seem like it should ever happen, but sometimes it does
        convolved = convolved[:len(audio)]
    return convolved