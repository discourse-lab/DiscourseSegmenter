#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Package providing discourse segmenter for Mate dependency graphs.

Attributes:
  __all__ (List[str]): list of sub-modules exported by this package
  __author__ (str): package's author
  __email__ (str): email of package's author
  __name__ (str): package's name
  __version__ (str): package version

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals

from dsegmenter.mateseg.dependency_graph import DependencyGraph, \
    read_trees, read_tok_trees, HEAD, ADDRESS, TOP_TAG_LABEL, \
    TOP_RELATION_LABEL
from dsegmenter.mateseg.matesegmenter import MateSegmenter, \
    trees2segs

##################################################################
# Intialization
__name__ = "mateseg"
__all__ = ["DependencyGraph", "HEAD", "ADDRESS",
           "TOP_TAG_LABEL", "TOP_RELATION_LABEL", "MateSegmenter",
           "read_trees", "read_tok_trees", "trees2segs"]
__author__ = "Andreas Peldszus"
__email__ = "peldszus at uni dash potsdam dot de"
__version__ = "0.2.0"
