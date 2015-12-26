#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

"""Main meta-package containing a collection of discourse segmenters.

Attributes:
  edseg (module): rule-based discourse segmenter for Mate dependency trees
  treeseg (module): auxiliary segmenter routines used by syntax-driven segmenters
  bparseg (module): machine-learning discourse segmenter for BitPar constituency trees
  __all__ (List[str]): list of sub-modules exported by this package
  __author__ (str): package's author
  __email__ (str): email of package's author
  __name__ (str): package's name
  __version__ (str): package version

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

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
