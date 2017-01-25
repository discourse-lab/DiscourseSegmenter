#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from pytest import fixture

import codecs
import os


##################################################################
# Constants
ENCODING = "utf-8"
DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")
MATESEG_INPUT1_FNAME = os.path.join(DATA_DIR, "mateseg.input.test1")
with codecs.open(MATESEG_INPUT1_FNAME, 'r', ENCODING) as ifile:
    MATESEG_INPUT1 = [iline for iline in ifile]


##################################################################
# Fixtures
@fixture(scope="module")
def mateseg_input1():
    return MATESEG_INPUT1
