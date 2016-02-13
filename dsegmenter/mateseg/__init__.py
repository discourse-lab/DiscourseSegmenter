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
from .dependency_graph import (DependencyGraph, read_deptree_file,
                               HEAD, DEPS, REL, TAG, ADDRESS,
                               TOP_TAG_LABEL, TOP_RELATION_LABEL)
from .segmentation_tree import read_segtree_file, generate_subtrees_from_forest

##################################################################
# Intialization
__name__ = "mateseg"
__all__ = ["DependencyGraph", "HEAD", "DEPS", "REL", "TAG", "ADDRESS",
           "TOP_TAG_LABEL", "TOP_RELATION_LABEL", "read_deptree_file",
           "read_segtree_file", "generate_subtrees_from_forest"]
__author__ = "Andreas Peldszus"
__email__ = "peldszus at uni dash potsdam dot de"
__version__ = "0.1.0"
