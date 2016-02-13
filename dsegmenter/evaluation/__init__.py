#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Package providing evaluation tools and metrics.

Attributes:
  __all__ (List[str]): list of sub-modules exported by this package
  __author__ (str): package's author
  __email__ (str): email of package's author
  __name__ (str): package's name
  __version__ (str): package version

"""

##################################################################
# Imports

from .align import align_tokenized_tree, AlignmentError
from .metrics import (metric_pk, metric_windiff, metric_pi_bed, metric_f1,
                      metric_lf1, metric_alpha_unit, metric_alpha_unit_untyped,
                      metric_kappa, avg, sigma)
from .segmentation import (get_typed_spans, get_untyped_spans,
                           get_typed_masses, get_untyped_masses,
                           typed_masses_to_spans,
                           get_typed_nonoverlapping_spans,
                           get_untyped_nonoverlapping_spans, get_confusions)

##################################################################
# Intialization
__name__ = "evaluation"
__all__ = ["align_tokenized_tree", "AlignmentError", "metric_pk",
           "metric_windiff", "metric_pi_bed", "metric_f1", "metric_lf1",
           "metric_alpha_unit", "metric_alpha_unit_untyped", "metric_kappa",
           "avg", "sigma", "get_typed_spans", "get_untyped_spans",
           "get_typed_masses", "get_untyped_masses", "typed_masses_to_spans",
           "get_typed_nonoverlapping_spans",
           "get_untyped_nonoverlapping_spans", "get_confusions"]
__author__ = "Andreas Peldszus"
__email__ = "peldszus at uni dash potsdam dot de"
__version__ = "0.1.0"
