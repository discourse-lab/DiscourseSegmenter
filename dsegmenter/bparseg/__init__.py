#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Package providing discourse segmenter for BitPar constituency trees.

Attributes:
  align (module): auxiliary routines for doing string alignment
  constants (module): constants specific to that package
  constituency_tree (module): proxy class for handling constituency syntax
                              trees
  bparsegmenter (module): class for segmenting syntax trees into discourse
                          units
  __all__ (list[str]): list of sub-modules exported by this package
  __author__ (str): package's author
  __email__ (str): email of package's author
  __name__ (str): package's name
  __version__ (str): package version

"""

##################################################################
# Imports
from __future__ import absolute_import

from dsegmenter.bparseg.constants import ENCODING, NO_PARSE_RE, WORD_SEP
from dsegmenter.bparseg.bparsegmenter import BparSegmenter, \
    read_trees, read_tok_trees, trees2segs
from dsegmenter.bparseg.constituency_tree import CTree, OP, OP_RE, CP, CP_RE

##################################################################
# Intialization
__name__ = "bparseg"
__all__ = ["ENCODING", "NO_PARSE_RE", "WORD_SEP", "BparSegmenter", "CTree",
           "OP", "OP_RE", "CP", "CP_RE", "read_trees", "read_tok_trees",
           "trees2segs"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
