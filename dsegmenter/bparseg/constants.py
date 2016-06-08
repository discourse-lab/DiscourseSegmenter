#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Module defining necessary constants for that package.

Attributes:
  ENCODING (str): default encoding used for strings
  NO_PARSE_RE (re): regexp that matches sentences for which no BitPar
    tree was generated
  WORD_SEP (re): regexp matching word delimiters

"""

##################################################################
# Imports
import re


##################################################################
# Constants
ENCODING = "utf-8"
NO_PARSE_RE = re.compile("""\s*No\s+parse\s+for\s*:\s*"([^\n]+)"$""",
                         re.IGNORECASE)
WORD_SEP = re.compile("\s+")
