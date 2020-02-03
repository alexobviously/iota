#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import warnings

# ---------------
# plot_audio
# Just a debug function to plot a wave graph of the given audio
# ---------------  
def plot_audio(fig,aud,_sr,title,sec):
    plt.figure(fig)
    plt.title(title)
    if sec:
        # Calculate the x axis values if it's going to be in seconds
        time = np.linspace(0, len(aud)/_sr, num=len(aud))
        plt.plot(time,aud)
    else:
        plt.plot(aud)

# ---------------
# plot_lines
# Adds vertical lines to the current plot
# (used to display where the events are taken from)
# ---------------
def plot_lines(fig,lines,colours,lw=1):
    _colours = ['r','k','c','y','b','g','m']
    for i in range(0,len(lines)):
        plt.axvline(x=lines[i],linewidth=lw,color=_colours[colours[i]%len(_colours)])

# ---------------
# plot_features
# Creates a scatter plot of the first two features, colour coded by group.
# ---------------
def plot_features(event_groups,features,num_groups):
    plt.figure(2)
    plt.title("Two features")
    _colours = ['r','k','c','y','b','g','m']
    for x in range(0,num_groups):
        plt.scatter(features[0,event_groups==x],features[1,event_groups==x],color=_colours[x%len(_colours)])

# ---------------
# plot_source_audio
# Creates a plot of the source audio file with every event drawn on, colour coded by group.
# Usually results in a solid block of colour though, unless spacing is particularly low.
# ---------------
def plot_source_audio(audio,sr,event_list,event_groups):
    plot_audio(1,audio,sr,'Source Audio + Event Groups',False)
    plot_lines(1,event_list,event_groups)

# ---------------
# plot_generated_audio
# Just plots the generated audio.
# ---------------
def plot_generated_audio(audio,sr):
    plot_audio(0,audio,sr,'Generated Audio',True)

# ---------------
# plot_dynamic_range
# Creates histograms of the dynamic range before and after compression.
# ---------------
def plot_dynamic_range(pre,post):
    plt.figure(3)
    plt.subplot(3,1,1)
    plt.title("Pre-compression dynamic range")
    plt.hist(pre,400,[0.,max(pre)])
    plt.subplot(3,1,2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    plt.title("Post-compression dynamic range")
    plt.hist(post,400,[0.,max(post)])

# ---------------
# show
# Just a wrapper
# --------------- 
def show():
    plt.show()