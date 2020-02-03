#!/usr/bin/env python

import numpy as np
import scipy.signal as sig
import scipy.fftpack as fftp
import audio as au

# ---------------
# grainstream
# A class comprising one stream of grains.
# ---------------
class grainstream:
    def __init__(self,offset,grain_size,num_grains,sample_rate):
        self.audio = np.empty([num_grains,grain_size])
        self.grains = []
        self.next_grain = 0
        self.need_update = False
        self.offset = offset
        self.grain_size = grain_size
        self.sample_rate = sample_rate
        self.grain_window = au.tukey(grain_size,0.1)

    # ---------------
    # extend
    # Adds a grain onto the audio array and applies any effects to it.
    # ---------------
    def extend(self,grain,effects,stats):
        global conv_count
        _audio = grain.get_audio()
        if effects:
            for e in effects:
                if e[0] == 'filter':
                    _audio = au.filter_audio(_audio,self.sample_rate,e[1],e[2],e[3],e[4])
                    stats.filterings += 1
                if e[0] == 'convolve' and self.next_grain>0:
                    stats.convolutions += 1
                    _max = np.amax(np.abs(_audio))
                    _audio = sig.fftconvolve(_audio,self.audio[self.next_grain-1],mode="same")
                    _audio = au.normalise(_audio,_max)*self.grain_window
                    nn = np.amax(np.abs(_audio))
        self.audio[self.next_grain] = _audio
        self.grains.append(grain)
        self.next_grain += 1
        stats.num_grains += 1
    # ---------------
    # pad
    # Small but super important. Makes each grain stream start at a slightly different offset,
    # in order to prevent strange resonances and allow a full textures.
    # ---------------
    def pad(self,_audio):
        return np.concatenate([np.zeros(self.offset),_audio,np.zeros(self.grain_size-self.offset)])
    
    # ---------------
    # smooth_audio
    # Currently unused but may bring it back as an effect at some point.
    # Blurs grains onto smooth_distance adjacent grains.
    # ---------------
    def smooth_audio(self,smooth_distance,smooth_level):
        smoothing_grains = []
        for i in range(0,len(self.audio)):
            sg = np.zeros(self.audio[0].size)
            for j in range(-smooth_distance-1,smooth_distance+1):
                if i+j >= 0 and i+j < len(self.grains) and j != 0:
                    sg += self.audio[i+j]*(abs(j)/float(smooth_distance))
            smoothing_grains.extend(sg)
        smoothing_grains = np.array(smoothing_grains)
        smoother = np.concatenate([np.zeros(self.offset),smoothing_grains.reshape(smoothing_grains.size)])
        self.audio = au.normalise(self.audio,1.0) + au.normalise(smoother,smooth_level)
        
    # ---------------
    # get_audio
    # Applies random panning to the grain stream, pads it and returns it.
    # ---------------   
    def get_audio(self):
        _audio = self.audio.reshape(self.audio.size)
        num_grains = self.audio.shape[0]
        pan_g = np.random.normal(0.,0.4,num_grains) # pan value per grain
        pan_a = np.repeat(pan_g,self.grain_size) # expands pan_g out to the audio domain
        pan_l = np.clip(-pan_a+1,0.,1.)
        pan_r = np.clip(pan_a+1,0.,1.)
        return [self.pad(_audio*pan_l),self.pad(_audio*pan_r)]
    
    # ---------------
    # get_length
    # Returns the length of the grain stream (with one extra grain length for padding).
    # ---------------
    def get_length(self):
        return self.audio.size + self.grain_size

# ---------------
# grain
# A class that stores the audio and features of one event.
# ---------------
class grain:
    def __init__(self,_audio,features):
        self.audio = _audio
        self.features = features

    def get_audio(self):
        return self.audio
# ---------------
# graingroup
# A class used to group grains classes.
# ---------------
class graingroup:
    def __init__(self):
        self.grains = []
    
    # ---------------
    # add_grain
    # Appends a grain to the grain group list.
    # ---------------
    def add_grain(self,_grain):
        self.grains.append(_grain)
    # ---------------
    # random_grain
    # Returns a random grain from the grain list
    # ---------------
    def random_grain(self):
        return self.grains[np.random.randint(0,len(self.grains))]