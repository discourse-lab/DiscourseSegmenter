#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Module providing methods for string alignment.

All of these methods take two iterables (can be either lists or strings) as
arguments.  The first iterable (L1) represents string or list to which
alignment should be done, another iterable (L2) represents string or list which
should be aligned.  Optionally, penalties for different types of modifications
(insertion, deletion, substitution) can be specified either as positional or as
keyword arguments.  These arguments should be functions which accept one (for
insertion or deletion) or two (for substitution) characters (or any list
elements) as arguments and return corresponding penalty scores for
modifications.  The output is another list which has length equal to the length
of L1, where each element is a list of indices of L2 corresponding to given L1
elements.

Example:
  nw_align("AGTACGCA", "TCGC")
    => [[], [], [0], [], [1], [2], [3], []]


this corresponds to the alignment

  AGTACGCA
    T CGC

Please also note that different algorithms may give different alignments for
tie cases.

.. moduleauthor: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import print_function

import sys


##################################################################
# Methods
def hb_align(s1, s2, insert=lambda c: -2,
             delete=lambda c: -2,
             substitute=lambda c1, c2: 2 if c1 == c2 else -1,
             offset=0, keep_deleted=False):
    """Align two iterables using Hirschberg alignment algorithm.

    Complexity: :math:`O(nm)` time; :math:`O(min(n, m))` space

    Args:
      s1 (iterable): iterable for alignment
      s2 (iterable): iterable which should be aligned
      insert (lambda): function returning penalty for insertion (default -2)
      delete (lambda): function returning penalty for deletion (default -2)
      substitute (lambda): function returning penalty for substitution
        (default -1)
      offset (int): add ``offset`` to each aligned index of second iterable
                   (this option is only needed for recursive alignmnet of
                   substrings or sublists)
      keep_deleted (bool): return indices of deleted words too

    Returns:
      list: indices of s2 corresponding to given positions of s1

    """
    # remember penalty functions
    penalties = [insert, delete, substitute]
    # determine length of both lists
    L1 = len(s1)
    L2 = len(s2)
    # return value will always be a list of L1 list elements
    ret = []
    # establish auxiliary variables
    mid1 = mid2 = 0
    # one easy case is when one of the iterables is empty
    if L1 == 0 or L2 == 0:
        ret = [[] for c in s1]
    # another easy case is when both iterables are identical
    elif s1 == s2:
        ret = [[i + offset] for i in xrange(L1)]
    # the trickier case, however, is when actual alignment should be done
    elif L1 == 1 or L2 == 1:
        ret = nw_align(s1, s2, *penalties, offset=offset)
    else:
        mid1 = L1 / 2

        ScoreL = _nw_score_(s1[:mid1], s2)
        ScoreR = _nw_score_([c for c in reversed(s1[mid1:])],
                            [c for c in reversed(s2)])
        mid2 = _partition_(ScoreL, [i for i in reversed(ScoreR)])

        ret += hb_align(s1[:mid1], s2[:mid2], *penalties,
                        offset=offset)
        ret += hb_align(s1[mid1:], s2[mid2:], *penalties,
                        offset=offset + mid2)
    return ret


def nw_align(s1, s2, insert=lambda c: -2,
             delete=lambda c: -2,
             substitute=lambda c1, c2: 2 if c1 == c2 else -1,
             offset=0, keep_deleted=False):
    """Align two iterables using Needleman-Wunsch algorithm.

    Complexity: :math:`O(nm)` time; :math:`O(nm)` space

    Args:
      s1 (iterable): iterable for alignment
      s2 (iterable): iterable which should be aligned
      insert (lambda): penalty for insertion (default -2)
      delete (lambda): penalty for deletion (default -2)
      substitute (lambda): penalty for substitution (default -1)
      offset (int): add ``offset`` to each aligned index of second iterable
                   (this option is only needed for recursive alignmnet of
                   substrings or sublists)
      keep_deleted (bool): return indices of deleted words too

    Returns:
      list: indices of s2 corresponding to given positions in s1

    """
    # create optimal matching matrix
    mtx = _make_matrix_(s1, s2, insert, delete, substitute)
    # decode best alignment using this matrix
    return _decode_matrix_(mtx, offset, keep_deleted)


##################################################################
# Private Methods
def _make_matrix_(s1, s2, insert, delete, substitute):
    """Create and populate optimal matching matrix of two strings.

    This method populates and returns a two dimensional alignment matrix of two
    strings `s1` and `s2`.  Cells of this matrix consist of two-tuples
    where first element is the optimal alignment score and second element is
    another two-tuple with indices of source element which yielded this score.

      s1 - first string to align
      s2 - second string to align
      insert - function giving penalty for inserting a char
      delete - function giving penalty for deleting a char
      substitute - function giving penalty for substituting a char1 with char2

    Returns:
      np.array: optimal matching matrix of two strings

    """
    # get lengths of both lists
    L1 = len(s1) + 1
    L2 = len(s2) + 1
    # create a matrix for storing scores and backtracking information.
    mtx = [[[None, None]] * L2 for c in xrange(L1)]
    # auxiliary variables for characters
    c1 = c2 = ''
    # auxiliary variables for iterators
    i = j = prev_i = prev_j = 0
    # iterator over the second string
    s1_it = xrange(1, L1)
    s2_it = xrange(1, L2)
    # individual scores for insertion, deletion, and substitution
    inscore = delscore = subscore = maxscore = 0
    # backtracking index
    bck_idx = ()
    # base case when both strings are empty
    mtx[0][0] = (0, None)
    # populate base case when string 1 is empty
    for j in s2_it:
        prev_j = j - 1
        bck_idx = (0, prev_j)
        delscore = mtx[0][prev_j][0] + delete(s2[prev_j])
        # remember deletion score computed from the cell below
        mtx[0][j] = (delscore, bck_idx)
    # iterate over the first string
    for i in s1_it:
        prev_i = i - 1
        # current character of the first string will be at position `i - 1`
        # (because all i's are actually shifted by one due to the base 0 case)
        c1 = s1[prev_i]
        # the 0-th index is the base case, when string 2 is exhausted
        mtx[i][0] = (mtx[prev_i][0][0] + delete(c1), (prev_i, 0))
        # iterate over the second string
        for j in s2_it:
            prev_j = j - 1
            # obtain character of the second string
            c2 = s2[prev_j]
            # compute different modification scores
            delscore = mtx[prev_i][j][0] + delete(c1)
            inscore = mtx[i][prev_j][0] + insert(c2)
            subscore = mtx[prev_i][prev_j][0] + substitute(c1, c2)
            # compute the maximum over three scores
            maxscore = max(delscore, inscore, subscore)
            # determine from which element the maximum score came
            if subscore == maxscore:
                bck_idx = (prev_i, prev_j)
            elif inscore == maxscore:
                bck_idx = (i, prev_j)
            else:
                bck_idx = (prev_i, j)
            # populate matrix cell with the maximum score and its source cell
            mtx[i][j] = (maxscore, bck_idx)
    return mtx


def _decode_matrix_(mtx, a_offset=0, a_keep=False):
    """Compute best alignment for s1 and s2 based on error matrix.

    Args:
      mtx (np.array): optimal matching matrix
      a_offset (int): offset for indices
      a_keep (bool): include indices of words the were deleted during edit

    Returns:
      list: indices of second iterable which provide best alignment to
        the first iterable

    """
    # matrix indices `i` and `j` will differ by one from the actual string
    # indices for strings `s1` and `s2`
    i = len(mtx) - 1
    j = len(mtx[0]) - 1
    # return value will be a list of length `len(s1)` whose elements in turn
    # will be lists of element indices of `s2`
    ret = [[] for c in xrange(i)]
    # backtracking indices
    prev_i = prev_j = 0
    while i > 0 or j > 0:
        # get backtracking indices, i.e. indices of matrix element which led to
        # the given cell
        prev_i, prev_j = mtx[i][j][1]
        if prev_j == j:
            # if `j` is the same, it means that the element from the first
            # list was deleted, then do nothing but simply check our sanity
            assert(prev_i != i)
        elif prev_i != i:
            # if neither `i` nor `j` are the same, this means full
            # correspondence
            ret[i - 1].insert(0, j - 1 + a_offset)
        # uncomment this, if you want to get a list of deleted elements for a
        # single position
        elif a_keep:
            # if `i` is the same or neither indices are the same, it means that
            # character from the second string was inserted (in the former
            # case) or substituted (in the latter case)
            if i >= len(ret):
                i = len(ret) - 1
            ret[i].insert(0, j - 1 + a_offset)
        # backtrack
        i, j = prev_i, prev_j
    # return calculated alignment list
    return ret


def _partition_(seq1, seq2):
    """Find a pair of elements in iterables seq1 and seq2 with maximum sum.

    Args:
      seq1 (iterable): with real values
      seq2 (iterable): with real values

    Returns:
      int: list index such that seq1[pos] + seq2[pos] is maximum

    """
    _sum_ = _max_ = _pos_ = float("-inf")
    for pos, ij in enumerate(zip(seq1, seq2)):
        _sum_ = sum(ij)
        if _sum_ > _max_:
            _max_ = _sum_
            _pos_ = pos
    return _pos_


def _nw_score_(s1, s2, insert=lambda c: -2,
               delete=lambda c: -2,
               substitute=lambda c1, c2: 2 if c1 == c2 else -1):
    """Compute Needleman Wunsch score for aligning two strings.

    This algorithm basically performs the same operations as Needleman Wunsch
    alignment, but is made more memory efficient by storing only two columns of
    the optimal alignment matrix.  As a consequence, no reconstruction is
    possible.

    Args:
      s1 (iterable): iterable to which we should align
      s2 (iterable): iterable to be aligned
      insert (lambda): function returning penalty for insertion (default -2)
      delete (lambda): function returning penalty for deletion (default -2)
      substitute (lambda): function returning penalty for substitution
        (default -1)

    Returns:
      : last column of optimal matching matrix

    """
    # lengths of two strings are further used for ranges, therefore 1 is added
    # to every length
    m = len(s1) + 1
    n = len(s2) + 1
    # score will be a two dimensional matrix
    score = [[0 for i in xrange(n)], [0 for i in xrange(n)]]
    # character of first and second string, respectively
    c1 = c2 = ''
    # iterator over the second string
    s2_it = xrange(1, n)
    # indices of current and previous column in the error matrix (will be
    # swapped along the way)
    crnt = 0
    prev = 1
    prev_j = 0
    # base case when the first string is shorter than second
    for j in s2_it:
        prev_j = j - 1
        score[crnt][j] = score[crnt][prev_j] + insert(s2[prev_j])
    # iterate over the first string
    for i in xrange(1, m):
        # swap current and previous columns
        prev, crnt = crnt, prev
        # get current character of the first string
        c1 = s1[i - 1]
        # calculate the base case when len = 0
        score[crnt][0] = score[prev][0] + delete(c1)
        for j in s2_it:
            prev_j = j - 1
            c2 = s2[prev_j]
            # current cell will be the maximum over insertions, deletions, and
            # substitutions applied to adjacent cells
            # substitution (covers cases when both chars are equal)
            score[crnt][j] = max(score[prev][prev_j] + substitute(c1, c2),
                                 # deletion
                                 score[prev][j] + delete(c1),
                                 # insertion
                                 score[crnt][prev_j] + insert(c2))
    # return last computed column of scores
    return score[crnt]
