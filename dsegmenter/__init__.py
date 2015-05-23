#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

"""
Main meta-package containing a collection of discourse segmenters

Subpackages:
edseg - rule-based discourse segmenter for Mate dependency trees
treeseg - auxiliary segmenter routines used by syntax-driven segmenters
bparseg - machine-learning discourse segmenter for BitPar constituency trees

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports

##################################################################
# Variables and Constants
__name__ = "dsegmenter"
__all__ = ["edseg", "bparseg", "treeseg"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
