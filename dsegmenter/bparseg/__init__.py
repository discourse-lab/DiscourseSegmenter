#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Package providing discourse segmenter for BitPar constituency trees.

Modules:
align - auxiliary routines for doing string alignment
constants - defines constants specific to that package
constituency_tree - proxy class for handling constituency syntax trees
bparsegmenter - class for segmenting syntax trees into dicsourse units

"""

##################################################################
# Imports
from constants import ENCODING, NO_PARSE_RE, WORD_SEP
from bparsegmenter import BparSegmenter, read_trees, read_segments, trees2segs
from constituency_tree import CTree

##################################################################
# Intialization
__name__ = "bparseg"
__all__ = ["ENCODING", "NO_PARSE_RE", "WORD_SEP", "BparSegmenter", "CTree", \
               "read_trees", "read_segments", "trees2segs"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
