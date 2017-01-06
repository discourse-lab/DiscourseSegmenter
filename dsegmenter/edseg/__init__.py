#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

"""Package providing rule-based discourse segmenter for CONLL trees.

Attributes:
  chunking (module): routines for internal clause segmentation
  clause_segmentation (module): utilities and classes for rule-based clause
    segmenter
  conll (module): interface for dealing with CONLL data
  data (module): data definitions and data reading routines
  edssegmenter (module): definition of rule-based discourse segmenter
  finitestateparsing (module): parsing routines based on finite-state
    mechanisms
  util (module): auxiliary match routines needed for rule matching

"""

##################################################################
# Imports
from __future__ import absolute_import

from dsegmenter.edseg.conll import CONLL
from dsegmenter.edseg.edssegmenter import EDSSegmenter

##################################################################
# Variables and Constants
__name__ = "edseg"
__all__ = ["CONLL", "EDSSegmenter"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.0.1"
