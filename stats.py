#!/usr/bin/env python

# ---------------
# stats
# Just a little structure to store some global run statistics.
# ---------------
class stats:
    def __init__(self):
        self.convolutions = 0
        self.filterings = 0
        self.num_grains = 0
        self.num_events = 0