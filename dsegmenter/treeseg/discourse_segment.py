#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Module providing Discourse Segment class.

Class:
DiscourseSegment - class representing discourse segment

@author = Wladimir Sdorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from .constants import ENCODING

from bisect import bisect_right

##################################################################
# Class
class DiscourseSegment(object):
    """
    Class representing discourse segment.

    Instance Variables:
    name - name of this segment
    leaves - list of words or other segments inside of this unit

    Methods:
    get_end - obtain position of last token in the list of leaves
    """

    def __init__(self, a_name = "", a_leaves = []):
        """
        Class constructor.

        @param a_name - name of discourse segment
        @param a_leaves - list of segment's child nodes (either words or other segments)
        """
        self.name = a_name
        self.leaves = a_leaves
        self.leaves.sort(key = lambda el: el[0] if el else -1)

    def get_end(self):
        """
        Obtain position of last token in the list of leaves.
        """
        if not self.leaves:
            return -1
        else:
            last_leaf = self.leaves[-1]
            if isinstance(last_leaf[-1], DiscourseSegment):
                return last_leaf[-1].get_end()
            else:
                return last_leaf[0]

    def insort(self, a_leaf):
        """
        Insert leaf in the list of leaves according to its position.

        @param a_leaf - leaf to be inserted

        @return void
        """
        ipos = bisect_right(self.leaves, a_leaf)
        inserted = False
        prev_pos = ipos - 1
        if prev_pos >= 0:
            prev_leaf = self.leaves[prev_pos]
            if isinstance(prev_leaf[-1], DiscourseSegment) and \
                    prev_leaf[-1].get_end() > a_leaf[0]:
                prev_leaf[-1].insort(a_leaf)
                inserted = True
        if not inserted:
            self.leaves.insert(ipos, a_leaf)

    def __len__(self):
        """
        Return number of elements in given segment.

        @return number of elements in segment
        """
        return len(self.leaves)

    def __nonzero__(self):
        """
        Return True if the given segment is not empty.

        @return True if segment is not empty
        """
        return bool(self.leaves)

    def __unicode__(self):
        """
        Return unicode representation of given segment.

        @return unicode string representing this object
        """
        ret = u"(" + unicode(self.name)
        for t in self.leaves:
            ret += u' ' + unicode(t[-1])
        ret += u" )"
        return ret

    def __str__(self):
        """
        Return utf-8 string representation of given segment.

        @return utf-8 string representing this object
        """
        return self.__unicode__().encode(ENCODING)

    def __repr__(self):
        """
        Return internal representation of given segment.

        @return internal representation of this segment
        """
        ret = '<' + self.__class__.__name__ + " at " + str(hex(id(self)))
        ret += u" name=" + repr(self.name)
        ret += u" leaves=" + repr(self.leaves) + ">"
        return ret
