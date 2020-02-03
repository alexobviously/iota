#!/usr/bin/env python

import argparse as ap
import sys
import numpy as np

# ---------------
# parser_error
# Just a function to throw an error and quit the program.
# ---------------
def parser_error(error):
    print "Error: %s" % error
    sys.exit()
    
# ---------------
# to_float
# Just a macro for returning a different error if float conversion fails.
# ---------------
def to_float(x,e):
    try:
        return float(x)
    except ValueError:
        parser_error(e)
        
# ---------------
# parameters
# Just a structure to make passing parameters around a bit less fragile.
# ---------------
class parameters:
    def __init__(self,infile,outfile,grain_size,grain_spacing,num_streams,num_groups,num_features,dzc,mode,modevars,fx,comp_thresh,comp_ratio,norm_level,fade_size,emptiness,debug):
        self.infile = infile
        self.outfile = outfile
        self.grain_size_ms = grain_size
        self.grain_size = (self.grain_size_ms*44100)/1000 # default value, set for certain after sample rate received though
        self.grain_spacing_ms = grain_spacing
        self.grain_spacing = (self.grain_spacing_ms*44100)/1000 # likewise
        self.num_streams = num_streams
        self.num_groups = num_groups
        self.num_features = num_features
        self.dzc = dzc
        self.mode = mode
        self.modevars = modevars
        self.fx = fx
        self.comp_thresh = comp_thresh
        self.comp_ratio = comp_ratio
        self.norm_level = norm_level
        self.fade_size = fade_size
        self.emptiness = emptiness
        self.debug = debug
        
# ---------------
# parse_args
# Parses all command line arguments and does a few basic sanitisations.
# ---------------
def parse_args():
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i","--infile",help="Input audio file")
    parser.add_argument("-o","--outfile",help="Audio file to output to")
    parser.add_argument("-g","--grainsize",help="Length of each grain, in ms",type=int,default=20)
    parser.add_argument("-c","--grainspacing",help="Spacing between each grain in analysis, in ms",type=int,default=200)
    parser.add_argument("-s","--numstreams",help="Number of grain streams",type=int,default=100)
    parser.add_argument("-r","--numgroups",help="Number of groups to use for clustering",type=int,default=5)
    parser.add_argument("-f","--numfeatures",help="Number of features to use for clustering",type=int,default=8)
    parser.add_argument("-z","--disablezerocrossings",help="Don't use zero-crossings as a clustering feature",action="store_true")
    parser.add_argument("-m","--mode",choices=["block","loop"],default="loop",help="Generator mode, read documentation for more information")
    parser.add_argument("-l","--numloops",type=int,default=3,help="Number of loops to use in loop mode")
    parser.add_argument("-p","--grouplength",type=float,default=2.0,help="Number of seconds each group should last in loop mode")
    parser.add_argument("-b","--blocks",nargs="*",help="Block parameters for blocks mode, in the form 'identifier start end fade', read documentation for more information")
    parser.add_argument("-e","--emptiness",type=float,default=0.2,help="Introduces an element of sparseness, block mode only, read documentation for more information")
    parser.add_argument("-x","--effects",nargs="*",help="Effect parameters, in the form 'lowpass/highpass/convolve identifier cutoff transition_bandwidth attenuation, read documentation for more information")
    parser.add_argument("-t","--compthresh",type=float,default=0.2,help="Threshold for compression, specifies a percentage that should be compressed at the top of the dynamic range, e.g. 0.1 compresses top 10 percent")
    parser.add_argument("-a","--compratio",type=float,default=2.5,help="Compression ratio")
    parser.add_argument("-n","--normlevel",type=float,default=0.9,help="Level to normalise to")
    parser.add_argument("-d","--fadesize",type=float,default=0.05,help="Size of the fade in and fade out, corresponds to the alpha value of a Tukey window")
    parser.add_argument("-u","--debug",choices=['0','1','2'],default=0,help="Debug level: 0 is off, 1 outputs some text, 2 outputs text and plots some useful graphs")
    
    args = parser.parse_args()
    infile = args.infile
    outfile = args.outfile
    grainsize = args.grainsize
    grainspacing = args.grainspacing
    numstreams = args.numstreams
    numgroups = args.numgroups
    numfeatures = args.numfeatures
    dzc = args.disablezerocrossings
    mode = args.mode
    looplength = (args.grouplength * 1000) - ((args.grouplength*1000) % grainsize)
    comp_thresh = args.compthresh
    comp_ratio = args.compratio
    norm_level = args.normlevel
    fade_size = args.fadesize
    emptiness = args.emptiness
    debug = int(args.debug)

    if infile == None:
        parser_error('No input file specified')
    if outfile == None:
        parser_error('No output file specified')
    if grainsize < 1:
        parser_error('Grain size must be at least 1ms')
    if grainsize < 1:
        parser_error('Grain spacing must be at least 1ms')
    if numstreams < 1:
        parser_error('Number of grain streams must be at least 1')
    if numgroups < 1:
        parser_error('Number of cluster groups must be at least 1')
    if numfeatures < 1:
        parser_error('Number of features must be at least 1')
    if comp_thresh < 0. or comp_thresh > 1.0:
        parser_error("Compression threshold must be between 0 and 1")
    if comp_ratio <= 0.:
        parser_error("Compression ratio must be at least 0.")
    if norm_level <= 0. or norm_level >= 1.0:
        parser_error("Normalisation level must be more than 0 and less than 1")
    if fade_size < 0. or fade_size > 1.0:
        parser_error("Fade size must be between 0 and 1")
    if emptiness < 0.:
        parser_error("Emptiness cannot be less than 0.0")
        
    if mode == "loop":
        if args.numloops < 1:
            parser_error("number of loops must be at least 1")
        if looplength < grainsize:
            parser_error("loop length must be at least one grain long (%.4f seconds)" % (grainsize/1000.))
        modevars = [args.numloops,int(looplength/grainsize)]
    elif mode =="block":
        blocks = args.blocks
        if blocks == None or len(blocks)%4 != 0:
            parser_error("invalid or empty block list")
        modevars = np.reshape(blocks,(len(blocks)/4,4)).tolist()
        unique_identifiers = len(set([x for x,y,z,w in modevars]))
        if unique_identifiers > numgroups:
            print "Warning: %d unique identifiers entered in block list. Number of clustering groups increased from %d to %d to accommodate." % (unique_identifiers,numgroups,unique_identifiers)
            numgroups = unique_identifiers
        for i in range(0,len(modevars)):
            _start = to_float(modevars[i][1],"%s is not a valid start time" % modevars[i][1])*1000
            _end = to_float(modevars[i][2],"%s is not a valid end time" % modevars[i][2])*1000
            modevars[i][1] = int((_start - (_start % grainsize))/grainsize)
            modevars[i][2] = int((_end - (_end % grainsize))/grainsize)
            modevars[i][3] = float(modevars[i][3])
            if modevars[i][1] < 0:
                parser_error("start times must be at least 0")
            if modevars[i][2] <= modevars[i][1]:
                parser_error("end times must be at least one grain length later then start times")
            if modevars[i][3] > 1 or modevars[i][3] < 0:
                parser_error("windowing parameter must be between 0 and 1")
    fx = []
    current_fx = []
    if args.effects != None:
        for x in args.effects:
            if x in ["lp","lowpass","hp","highpass","cv","convolve"]:
                if current_fx != []:
                    fx.append(current_fx)
                current_fx = [x]
            else:
                if current_fx == []:
                    parser_error("Invalid effect type %s" % x)
                # identifier
                if len(current_fx) == 1:
                    current_fx.append(x)
                    continue
                # strength
                if len(current_fx) == 2:
                        if float(x)<=0 or float(x)>1.0:
                            parser_error("Effect strength must be between 0.0 and 1.0")
                        else:
                            current_fx.append(float(x))
                            continue
                if current_fx[0] in ["cv","convolve"]:
                    if len(current_fx) >= 3:
                        print "Warning: Too many parameters provided for convolve, parameter %s ignored" % x
                elif current_fx[0] in ["lp","lowpass","hp","highpass"]:
                    if len(current_fx) > 5:
                         print "Warning: Too many parameters provided for %s, parameter %s ignored" % (current_fx[0],x)
                    else:
                        if len(current_fx) == 3 and (float(x)<1 or float(x)>20000):# cutoff
                            parser_error("Cutoff must be between 1.0 and 20000.0 Hz")
                        elif len(current_fx) == 4 and (float(x)<1 or float(x)>20000):# transition band width
                            parser_error("Transition band width must be between 1.0 and 20000.0 Hz")
                        elif len(current_fx) == 5 and (float(x)<1 or float(x)>200):#attenuation
                            parser_error("Attenuation must be between 1.0 and 200.0 dB")
                        current_fx.append(float(x))
        fx.append(current_fx)
    unique_identifiers = []
    if len(fx) > 0:
        for x in fx:
            if len(x) == 1:
                parser_error("Missing identifier for %s" % x[0])
            else:
                if not x[1] in unique_identifiers:
                    unique_identifiers.append(x[1])
            if len(x) == 2:
                parser_error("Missing strength for %s" % x[0])
            if x[0] in ["lp","lowpass","hp","highpass"]:
                if len(x) == 3:
                    x.append(np.random.rand()*10000+2000) # random cutoff between 2000 and 12000 Hz
                if len(x) == 4:
                    x.append(np.random.rand()*800+200) # random transition band width between 200 and 1000 Hz
                if len(x) == 5:
                    x.append(60.0)
            if x[0] == "lp":
                x[0] = "lowpass"
            if x[0] == "hp":
                x[0] = "highpass"
            if x[0] == "cv":
                x[0] = "convolve"
        if len(unique_identifiers) > numfeatures:
            print "Warning: %d unique identifiers entered in effects list. Number of clustering features increased from %d to %d to accommodate." % (len(unique_identifiers),numfeatures,len(unique_identifiers))
            numfeatures = len(unique_identifiers)
            
    params = parameters(infile,outfile,grainsize,grainspacing,numstreams,numgroups,numfeatures,dzc,mode,modevars,fx,comp_thresh,comp_ratio,norm_level,fade_size,emptiness,debug)
    return params
    