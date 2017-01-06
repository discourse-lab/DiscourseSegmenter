#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Created on 03.01.2015

@author: Andreas Peldszus

"""

##################################################################
# Imports
from nltk.tree import Tree
import codecs


##################################################################
# Variables and Constants

segtree_leaf_counter = 0


##################################################################
# Methods
def prefix_number_seg_token(s):
    """adds an number prefix to a token string"""
    global segtree_leaf_counter
    r = (segtree_leaf_counter, s)
    segtree_leaf_counter += 1
    return r


def generate_subtrees_from_forest(forest):
    """yields all subtress of a forest of trees"""
    for tree in forest:
        for subtree in tree.subtrees():
            yield subtree


def read_segtree_file(fn):
    """reads a string representing a discourse tree (from the seg.
       annotation) and returns a list of its child tree objects"""
    with codecs.open(fn, 'r', 'utf-8') as f:
        s = f.read()
        text_tree = Tree.fromstring(s, read_leaf=prefix_number_seg_token)
        return [segment for segment in text_tree]
