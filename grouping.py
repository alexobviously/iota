#!/usr/bin/env python

import numpy as np
from scipy.io import wavfile as wav
import scipy.signal as sig
import scipy.fftpack as fftp
from scipy import log10, where
from scipy.cluster.vq import kmeans2 as kmeans

import grainstream as gs
import audio as au
import interface
# ---------------
# select_events
# Steps through an audio file at fixed intervals and adds each position to an event list.
# This is just one possibility really. It could be using transient detection or something instead. 
# --------------- 
def select_events(audio,ev_spacing,grain_size):
    _events = []
    i = 0
    biggest_gap = max(ev_spacing,grain_size)
    while i <= audio.size-biggest_gap:
        _events.append(i)
        i += ev_spacing
    return _events
# ---------------
# spectral_features
# Selects a random set of spectral features and calculates them for every event passed to it.
# ---------------
def spectral_features(audio,_events,ev_spacing,n_features=20,featurewidth=16):
    print "Selecting %d random spectral features.." % n_features
    feature_bins = np.random.randint(featurewidth/2,(ev_spacing/8),n_features)
    _features = np.zeros((n_features,len(_events)))
    ev_window = sig.hann(ev_spacing)
    for i in range(0,len(_events)):
        _ev = audio[_events[i]:_events[i]+ev_spacing]
        # Calculate spectrogram for the event
        _fftevent = _ev*ev_window
        mags = abs(fftp.rfft(_fftevent,ev_spacing))
        mags = 20*log10(mags) # dB
        # Calculate each feature for this event
        for j in range(0,n_features):
            _features[j][i] = abs(np.mean(abs(mags[(feature_bins[j]-featurewidth/2):(feature_bins[j]+featurewidth/2)])))
    return _features
# ---------------
# zero_crossings
# Calculates the zero-crossing count for every event and uses it to get an approximation
# of the fundamental frequency.
# ---------------
def zero_crossings(audio,_events,ev_spacing):
    _features = np.zeros((1,len(_events)))
    for i in range(0,len(_events)):
        _ev = audio[_events[i]:_events[i]+ev_spacing]
        zerocrossings = len(where(_ev[:-1] * _ev[1:] < 0)[0])
        f = zerocrossings*44100/ev_spacing
        _features[0][i] = f
    return _features
# ---------------
# cluster
# Runs k-means clustering on a set of features.
# ---------------
def cluster(_features,n_groups=5,iterations=30):
    print "Clustering.."
    return kmeans(np.transpose(_features),n_groups,minit='points',iter=30)
# ---------------
# group_events
# Selects events from some source audio, extracts features from them and groups them by clustering.
# ---------------
def group_events(audio,params):
    grain_size,spacing,no_zc,num_groups,num_features = params.grain_size,params.grain_spacing,params.dzc,params.num_groups,params.num_features
    event_list = select_events(audio,spacing,grain_size)
    # If zero crossings are disabled, use x spectral features, otherwise use x-1 and make zc the first feature.
    features = au.normalise(spectral_features(audio,event_list,spacing,num_features-(1-no_zc),16),1.0)
    if not no_zc:
        frequencies = au.normalise(zero_crossings(audio,event_list,spacing),1.0)
        features = np.concatenate((frequencies,features))
    [centroids,event_groups] = cluster(features,num_groups)
    grain_window = au.tukey(grain_size,0.1)
    events = []
    gg = []
    for i in range(0,num_groups):
        events.append([])
        gg.append(gs.graingroup())
    for i in range(0,len(event_list)):
        events[event_groups[i]] = np.append(events[event_groups[i]],int(event_list[i]))
        gr = gs.grain(audio[int(event_list[i]):int(event_list[i])+grain_size]*grain_window,features[:,i])
        gg[event_groups[i]].add_grain(gr)
    return [gg,event_list,event_groups,features]