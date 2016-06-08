#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""
Module providing class for handling constituency syntax trees.

Attributes:
  OP (str): special token used to substitute opening parentheses
  OP_RE (re): regexp for matching opening parentheses
  CP (str): special token used to substitute closing parentheses
  CP_RE (re): regexp for matching closing parentheses

.. moduleauthor:: Wladimir Sidorenko

"""

##################################################################
# Imports
from .constants import NO_PARSE_RE, WORD_SEP, ENCODING

import codecs
import nltk.tree
import nltk
import sys
import re

##################################################################
# Constants
OP = "-OP-"
OP_RE = re.compile(r"\\\(")
CP = "-CP-"
CP_RE = re.compile(r"\\\)")


##################################################################
# Classes
class Tree(nltk.tree.ParentedTree):
    """Direct subclass of nltk.tree.ParentedTree providing hashing.

    This class extends its parent by an additional method :meth:`__hash__`,
    which uses the standard :meth:`id` method and allows the objects to be
    stored in hashes, and also overwrites the method :meth:`prnt_label`,
    returning the label of the parent tree

    """

    def __init__(self, *args):
        """Class constructor (simply delegates to super-class).

        Args:
          args (list): arguments which should be passed to the parent

        """
        if len(args) == 0:
            super(Tree, self).__init__("", [])
        else:
            super(Tree, self).__init__(*args)

    def __hash__(self):
        """Return id of this object.

        """
        return id(self)

    def prnt_label(self):
        """Return label of this object.

        Returns:
          str: label of parent node or empty string if no parent exists

        """
        if self._parent:
            return self._parent.label()
        return ""


##################################################################
class CTree(Tree):
    """Class for reading and modifying constituency trees.

    This class subclasses :class:`Tree`.

    This class extends its parent by one additional public class method
    :meth:`parse_lines()`

    """

    @classmethod
    def parse_lines(cls, a_lines, a_one_per_line=False):
        """Parse input lines and return list of BitPar trees.

        Args:
          a_lines (list): decoded lines of the input file
          a_one_per_line (bool): flag indicating whether file is in one
                         sentence per line format

        Yields:
          constituency trees

        """
        lines = []
        itree = None
        imatch = None
        for iline in a_lines:
            iline = iline.strip()
            imatch = NO_PARSE_RE.match(iline)
            if imatch or not iline:
                if lines:
                    yield Tree.fromstring(u'\t'.join(lines))
                    del lines[:]
                if imatch:
                    yield Tree("TOP", imatch.group(1).split())
            else:
                if a_one_per_line:
                    try:
                        itree = Tree.fromstring(iline)
                        yield itree
                    except ValueError:
                        for seg in cls._get_segments(iline):
                            yield Tree.fromstring(seg)
                else:
                    lines.append(CP_RE.sub(CP, OP_RE.sub(OP, iline)))
        if lines:
            yield Tree.fromstring(u'\t'.join(lines))

    @classmethod
    def _get_segments(cls, a_line):
        """Split line into separate segments.

        Args:
          a_line (str): line to be split

        Returns:
          list: segments

        """
        seg = ""
        ob = 0
        segments = []
        max_len = len(a_line) - 1
        for i, c in enumerate(a_line):
            if c == "(" and i < max_len and not WORD_SEP.match(a_line[i + 1]):
                ob += 1
            elif c == ")":
                assert ob > 0, \
                    "Unmatched closing bracket in line" + repr(a_line)
                ob -= 1
            seg += c
            if ob == 0 and not WORD_SEP.match(seg):
                segments.append(seg)
                seg = ""
        assert ob == 0, "Unmatched opening bracket in line" + repr(a_line)
        return segments

    def __init__(self):
        """Class constructor.

        """
        pass
