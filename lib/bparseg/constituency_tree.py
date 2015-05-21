#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""
Module providing class for handling constituency syntax trees.

Classes:
Tree - meta-subclass of NLTK tree which allows hashing
CTree - interface for handling constituency trees

@author: Wladimir Sidorenko
"""

##################################################################
# Imports
from cmp_seg_bitpar import NO_PARSE_RE, WORD_SEP

import nltk.tree
import nltk
import sys
import re

##################################################################
# Classes
class Tree(nltk.tree.ParentedTree):
    """
    Direct subclass of nltk.tree.ParentedTree providing hashing.

    This class extends its parent by two additional methods:
    __hash__() - which uses the standard id() method and makes
    NLTK trees prnt_label() - return label of the parent tree
    """

    def __init__(self, *args):
        """
        Class constructor (simply delegates to super-class).

        @param args - list of arguments which should be passed to the parent

        """
        if len(args) == 0:
            super(Tree, self).__init__("", [])
        else:
            super(Tree, self).__init__(*args)

    def __hash__(self):
        """
        Return id of this object.
        """
        return id(self)

    def prnt_label(self):
        """
        Return label  of this object.

        @return label of parent node or empty string if no parent exists
        """
        if self._parent:
            return self._parent.label()
        return ""

##################################################################
class CTree(Tree):
    """
    Class for reading and modifying constituency trees.

    This class subclasses the Tree class.

    This class extends its parent by one additional public class method:
    parse_file - parse input file and return list of constituency trees

    """

    @classmethod
    def parse_file(cls, a_fname, a_encoding = "utf-8", a_one_per_line = False):
        """
        Parse input file and return list of BitPar trees.

        @param a_fname - name of the input file
        @param a_encoding - input file encoding
        @param a_one_per_line - flag oindicating whether file is in one
                         sentence per line format

        @return list of constituency trees
        """
        ret = []
        lines = []
        imatch = None
        with open(a_fname) as ifile:
            for iline in ifile:
                iline = iline.decode(a_encoding).strip()
                imatch = NO_PARSE_RE.match(iline)
                if imatch or not iline:
                    if lines:
                        ret.append(Tree.fromstring(u'\t'.join(lines)))
                        del lines[:]
                    if imatch:
                        ret.append(Tree("TOP", imatch.group(1).split()))
                else:
                    if a_one_per_line:
                        try:
                            ret.append(Tree.fromstring(iline))
                        except ValueError:
                            ret.append([Tree.fromstring(seg) for seg in cls._get_segments(iline)])
                    else:
                        lines.append(iline)
            if lines:
                ret.append(Tree.fromstring(u'\t'.join(lines)))
        return ret

    @classmethod
    def _get_segments(cls, a_line):
        """
        Split line into separate segments.

        @param a_line - line to be split

        @return list of segments
        """
        seg = ""
        ob = 0
        segments = []
        max_len = len(a_line) - 1
        for i, c in enumerate(a_line):
            if c == "(" and i < max_len and not WORD_SEP.match(a_line[i + 1]):
                ob += 1
            elif c == ")":
                assert ob > 0, "Unmatched closing bracket in line" + repr(a_line)
                ob -= 1
            seg += c
            if ob == 0 and not WORD_SEP.match(seg):
                segments.append(seg)
                seg = ""
        assert ob == 0, "Unmatched opening bracket in line" + repr(a_line)
        return segments

    def __init__(self):
        """
        Class constructor.

        """
        pass
