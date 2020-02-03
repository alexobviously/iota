#!/usr/bin/env python

import numpy as np
import grainstream as gs
import audio as au
import interface

# ---------------
# effects_manager
# Decides whether grains should have effects applied to them.
# ---------------
class effects_manager:
    # ---------------
    # __init__
    # Apart from initialising structure contents, also creates a list of unique FX identifiers.
    # ---------------
    def __init__(self,fx,features):
        self.filter_list = ['lowpass','highpass'] # just so it's not assigned a billion times
        self.fx = fx
        self.fx_features = np.random.permutation(range(0,features.shape[0]))
        self.fx_identifiers = []
        self.fx_distributions = []
        for x in self.fx:
            if not x[1] in self.fx_identifiers:
                feat_num = len(self.fx_identifiers)
                self.fx_identifiers.append(x[1])
                feat_sorted = np.sort(features[self.fx_features[feat_num]])
                dist_len = int(len(feat_sorted) - len(feat_sorted) * x[2])
                self.fx_distributions.append(feat_sorted[dist_len:])
                
    # ---------------
    # grain_fx
    # Decides whether a grain should have any effects applied to it.
    # ---------------
    def grain_fx(self,grain):
        effects = []
        for f in self.fx:
            ident_num = self.fx_identifiers.index(f[1])
            if grain.features[self.fx_features[ident_num]] >= np.random.choice(self.fx_distributions[ident_num]):
                if f[0] in self.filter_list:
                    effects.append(['filter',f[0],f[3],f[4],f[5]])
                elif f[0] == "convolve":
                    effects.append(['convolve'])
        return effects

# ---------------
# group_loop
# Loops through every grain group a specified number of times and interpolates between them.
# ---------------
def group_loop(sample_rate,params,grain_groups,features,stats):
    streams = []
    num_repeats,num_grains = params.modevars
    fx,num_streams,num_groups,grain_size = params.fx,params.num_streams,params.num_groups,params.grain_size
    
    if len(fx) > 0:
        fx_man = effects_manager(fx,features)

    for j in range(0,num_streams):
        print "Generating grain stream %d/%d.." % (j+1,num_streams)
        streams.append(gs.grainstream((grain_size/num_streams)*j,grain_size,num_grains*num_groups*num_repeats,sample_rate))
        for x in range(0,num_groups*num_repeats):
            for i in range(0,num_grains):
                if np.random.randint(0,num_grains)<=i:
                    g = (x+1) % num_groups
                else:
                    g = x % num_groups
                grain = grain_groups[g].random_grain()
                if len(fx) > 0:
                    effects = fx_man.grain_fx(grain)
                else:
                    effects = []
                streams[j].extend(grain,effects,stats)
    return streams

# ---------------
# block_generator
# Generates blocks of audio from grain groups depending on user input.
# ---------------
def block_generator(sample_rate,params,grain_groups,features,stats):
    num_streams,grain_size,emptiness,block_list,fx = params.num_streams,params.grain_size,params.emptiness,params.modevars,params.fx
    empty_grain = gs.grain(np.zeros(grain_size)+0.0000001,[]) # avoid divide by zero
    # parse user input and find unique identifiers to match to groups
    unique_identifiers = list(set([x for x,y,z,w in block_list]))
    num_groups = len(unique_identifiers)
    num_grains = max([z for x,y,z,w in block_list])
    # generate probability distributions across the entire audio file for each group
    group_dist = np.zeros([num_groups+1,num_grains])
    for id,_start,_end,alpha in block_list:
        group = unique_identifiers.index(id)
        group_dist[group,_start:_end] += au.tukey(_end-_start,alpha)
    group_dist[num_groups,:] = emptiness
    
    if len(fx) > 0:
        fx_man = effects_manager(fx,features)
    
    streams = []
    for j in range(0,num_streams):
        print "Generating grain stream %d/%d.." % (j+1,num_streams)
        streams.append(gs.grainstream((grain_size/num_streams)*j,grain_size,num_grains,sample_rate))
        for i in range(0,num_grains):
            # apply a bit of randomness and select the group with the highest probability
            r = np.random.rand(num_groups+1) * group_dist[:,i]
            _r = max(r)
            if _r == 0.:
                streams[j].extend(empty_grain,[],stats)
            else:
                g = np.where(r==_r)[0][0] # look up group by index
                if g == num_groups: # last group is emptiness
                    streams[j].extend(empty_grain,[],stats) # duplicate code, I know, but it's faster this way (avoids where calls)
                else:
                    grain = grain_groups[g].random_grain()
                    if len(fx) > 0:
                        effects = fx_man.grain_fx(grain)
                    else:
                        effects = []
                    streams[j].extend(grain,effects,stats)
    return streams
            