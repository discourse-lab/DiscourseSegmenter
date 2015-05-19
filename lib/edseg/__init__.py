#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

"""
Package providing rule-based discourse segmenter for CONLL dependency trees.

Constants:

Classes:
CONLL - custom class for parsing and printing CONLL trees
EDSSegmenter - class for doing rule based discourse segmentation

Exceptions:

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from conll import CONLL
from edssegmenter import EDSSegmenter

##################################################################
# Variables and Constants
__name__ = "edseg"
__all__ = ["CONLL", "EDSSegmenter"]
