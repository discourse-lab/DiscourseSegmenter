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
from constants import ENCODING
from BparSegmenter

##################################################################
# Intialization
__name__ = "bparseg"
__all__ = ["ENCODING", "BparSegmenter"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
