#!/usr/bin/env python

#===============================================================================
# IOTA
# ---------------
# Granular synthesis engine / Individual Project
# by Alex Baker
# 1057758 @ The University of Huddersfield
#===============================================================================

import numpy as np
import scipy.signal as sig
import scipy.fftpack as fftp
from scipy import log10
from scipy.cluster.vq import kmeans2 as kmeans
import sys

import grainstream as gs
import audio as au
import grouping as grp
import generator as gen
import interface
import stats as st

stats = st.stats()
params = interface.parse_args()
[source_audio,sample_rate,source_length] = au.read_audio(params.infile)
params.grain_size = (sample_rate*params.grain_size_ms)/1000
params.grain_spacing = (sample_rate*params.grain_spacing_ms)/1000
[grain_groups,event_list,event_groups,features] = grp.group_events(source_audio,params)
if params.debug > 0:
    stats.num_events = len(event_list)

if params.mode=='loop':
    streams = gen.group_loop(sample_rate,params,grain_groups,features,stats)
elif params.mode=='block':
    streams = gen.block_generator(sample_rate,params,grain_groups,features,stats)

print "Mixing down.."
output_audio = au.post_process(streams,params)
au.write_audio(params.outfile,sample_rate,output_audio)

if params.debug>0:
    print "Run stats:"
    print "  Number of events: %d" % stats.num_events
    print "  Number of grains: %d" % stats.num_grains
    print "  Number of effect convolutions: %d" % stats.convolutions
    print "  Number of filter uses: %d" % stats.filterings
    if params.debug>1:
        import plotting as pl
        print "Plotting.."
        pl.plot_features(event_groups,features,params.num_groups)
        pl.plot_source_audio(source_audio,sample_rate,event_list,event_groups)
        pl.plot_generated_audio(output_audio,sample_rate)
        pl.show()