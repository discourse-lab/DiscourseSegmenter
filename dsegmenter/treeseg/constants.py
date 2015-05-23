#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Module defining constants used in this package.

Constants:
DEFAULT_SEGMENT - default name for fallback segments
ENCODING - default encoding to be used for in- and output streams
GREEDY - only put words into segment which are adjacent to the segment
         initiating node
GENEROUS - put all words into segment which are lie between the root node and
         its outermost dependants
"""

##################################################################
# Constants
DEFAULT_SEGMENT = "HS"
DEPENDENCY = 1
CONSTITUENCY = 2
ENCODING = "utf-8"
GENEROUS = 1
GREEDY = 0
