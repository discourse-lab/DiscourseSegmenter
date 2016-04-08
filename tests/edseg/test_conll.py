#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import

from dsegmenter.edseg.conll import CONLLWord

from pytest import fixture
from unittest import TestCase


##################################################################
# Variables and Constants
CW_STR = "1	Das	_	der	_	ART	_	" \
         "nom|sg|neut	-1	3	_	NK	_	_"
CW1 = CONLLWord()
CW2 = CONLLWord(CW_STR)


##################################################################
# Test Classes
class TestCONLLWord(TestCase):

    @fixture(autouse=True)
    def set_feature(self):
        self.cw1 = CW1
        self.cw2 = CW2

    def test_init(self):
        assert self.cw1
        assert self.cw2

    def test_parse_line(self):
        self.cw1.parse_line(CW_STR)

    def test_contains(self):
        # issue #3
        assert 'feats' not in self.cw2
