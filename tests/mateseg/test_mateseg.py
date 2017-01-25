#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from dsegmenter.mateseg import read_trees
from dsegmenter.mateseg.matesegmenter import MateSegmenter

from pytest import fixture
from unittest import TestCase


##################################################################
# Variables and Constants
TEST1_OUT = "(HS (FRE Im Gegenteil : ) Ein Abbau weniger produktiver" \
    " Arbeitsplätze etwa würde zunächst einmal die" \
    " Arbeitslosigkeit erhöhen , die staatlichen Aufwendungen" \
    " steigern (HSF und die privaten Ausgaben verringern . ) )"


##################################################################
# Test Classes
class TestMateSegmenter(TestCase):

    @fixture(autouse=True)
    def set_vars(self, mateseg_input1):
        self.segmenter = MateSegmenter()
        self.input1 = [t for t in read_trees(mateseg_input1)]

    def test_segment_sentence_0(self):
        segments = self.segmenter.segment(self.input1)
        assert len(segments) == 1
        assert unicode(segments[0][-1]).strip() == TEST1_OUT
