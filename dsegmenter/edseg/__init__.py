#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

"""
Package providing rule-based discourse segmenter for CONLL dependency trees.

Modules:
chunking - routines for internal clause segmentation
clause_segmentation - rule-based clause segmenter
conll - interface for dealing with CONLL data
data - data definitions and data reading routines
edssegmenter - definition of rule-based discourse segmenter
finitestateparsing - parsing routines based on finite-state mechanisms
util - auxiliary match routines needed for rule matching

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from .conll import CONLL
from .edssegmenter import EDSSegmenter

##################################################################
# Variables and Constants
__name__ = "edseg"
__all__ = ["CONLL", "EDSSegmenter"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
