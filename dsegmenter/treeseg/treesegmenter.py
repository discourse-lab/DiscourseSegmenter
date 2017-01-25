#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""
Module for converting parse trees to discourse segments.

Class:
TreeSegmenter - class for converting parse trees to discourse segments

@author = Uladzimir Sidarenka
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsegmenter.common import WORD, REL, DEPS, TAG
from dsegmenter.treeseg.constants import GREEDY, GENEROUS, DEPENDENCY, \
    CONSTITUENCY
from dsegmenter.treeseg.discourse_segment import DiscourseSegment

##################################################################
# Constants
NO_MATCH_STRING = "NONE"


##################################################################
# Class
class TreeSegmenter(object):
    """Class for converting parse trees to discourse segments.

    Attributes:
      decfunc: decision function
      segment: public function for doing segmentation
      type: type of trees to be processed

    """

    def __init__(self, a_decfunc=None, a_type=DEPENDENCY):
        """
        Class constructor.
        """
        # set default decision function, if none was specified
        self.decfunc = a_decfunc

        # set default segmentation function, if none was specified
        if a_type == DEPENDENCY:
            self.segment = self._dg_segment
            if self.decfunc is None:
                self.decfunc = self._dg_decfunc
        elif a_type == CONSTITUENCY:
            self.segment = self._cnst_segment
            if self.decfunc is None:
                self.decfunc = self._cnst_decfunc
        else:
            raise RuntimeError("Invalid tree type specified for tree"
                               " segmenter: '{:s}'".format(a_type))

    def _dg_segment(self, a_tree, a_predict=None,
                    a_root_idx=0, a_children=[],
                    a_word_access=lambda x: x, a_strategy=GREEDY):
        """Extract discourse segments from dependency parse trees.

        Args:
          a_tree (nltk.parse.dependencygraph): parse tree which should be
            processed
          a_predict (lambda): prediction function
          a_root_idx (int): index of the root node in the list of tree nodes
          a_children (list[int]): index of the child nodes
          a_word_access (lambda): a function for accessing the token string for
                         more complex, structured tokens
          a_strategy (int): flag for handling missing and non-projective edges
            (GREEDY means that only adjacent descendants of the root node will
            be put into a segment, if the root initiates one; GENEROUS means
            that all words between the root and its right- and left-most
            dependencies will be put into one segment disregarding the actual
            structure of the dependency tree)

        Returns:
          list: discourse segments

        """
        a_ret = []
        dec = None             # decision
        if a_predict is None:
            a_predict = self.decfunc
        if a_children:
            children = a_children
        elif a_root_idx is None:
            children = []
        else:
            children = [a_root_idx]

        while children:
            ipos = children.pop(0)
            inode = a_tree.nodes[ipos]
            word = a_word_access(inode[WORD]) if WORD in inode else ""
            deps = a_tree.get_dependencies_simple(ipos)
            dec = a_predict(inode, a_tree)
            if dec is None or dec == NO_MATCH_STRING:
                # if the node did not start any new segment on its own, then we
                # add it to the top-most segment in the list
                if word is not None:
                    a_ret.append((ipos, word))
                children[:0] = deps
            else:
                # recurse
                leaves = self.segment(
                    a_tree, a_predict=a_predict, a_root_idx=None,
                    a_children=deps, a_word_access=a_word_access,
                    a_strategy=a_strategy)
                dseg = DiscourseSegment(a_name=dec, a_leaves=leaves)

                if word is not None:
                    dseg.insort((ipos, word))
                # if strategy is GREEDY, only leave nodes which are adjacent to
                # the root
                if a_strategy == GREEDY:
                    outleaves = self._extract_nonadjacent(dseg, ipos)
                    a_ret += outleaves
                a_ret.append((dseg.leaves[0][0] if dseg.leaves else -1, dseg))
        # for GENEROUS strategy, do some post-processing
        a_ret.sort(key=lambda el: el[0])
        if a_strategy == GENEROUS:
            a_ret[:] = self._unite_nonadjacent(a_ret)
        return a_ret

    def _dg_decfunc(self, a_node, a_tree):
        """Make a prediction whether given node initiates a segment.

        Args:
          a_node (dict): parse node to be analyzed
          a_tree (nltk.parse.dependencygraph): tree of analyzed node

        Returns:
          str or None: name of discourse segment or None

        """
        chnode = None
        chtag = ""
        irel = a_node.get(REL, "UNK")
        ideps = a_node.get(DEPS, [])
        if irel == "TOP":
            return "HS"
        else:
            for chpos in ideps:
                chnode = a_tree.nodelist[chpos]
                chtag = chnode.get(TAG, "UNK")
                if chtag == "KOUS":
                    return "SUB"
                elif chtag == "PRELS":
                    return "ARR"
            return None

    def _cnst_segment(self, a_tree, a_ret, a_predict=None, a_start=0):
        """Extract discourse segments from constitutency parse trees.

        Args:
          a_tree (nltk.parse.dependencygraph): parse tree which should
            be processed
          a_ret (list): target list which should be populated with segments
          a_predict (lambda): prediction function
          a_start (int): starting index of tokens

        Returns:
          list: discourse segments

        """
        if a_predict is None:   # find appropriate decision function
            a_predict = self.decfunc

        if isinstance(a_tree, basestring):
            a_ret.append((a_start, a_tree))
            a_start += 1
        else:
            dec = a_predict(a_tree)  # make decision about the tree
            if dec is None:
                for ch in a_tree:
                    a_start = self.segment(ch, a_ret, a_predict, a_start)
            else:
                # create a new segment
                dseg = DiscourseSegment(a_name=dec, a_leaves=[])
                # add tree leaves to the segment
                for ch in a_tree:
                    a_start = self.segment(ch, dseg.leaves, a_predict, a_start)
                # append the new segment to the tree
                if dseg.leaves and len(dseg.leaves[0]):
                    a_ret.append((dseg.leaves[0][0], dseg))
        return a_start

    def _cnst_decfunc(self, a_tree):
        """Make a prediction whether given parse tree initiates a segment.

        Args:
         a_tree (nltk.parse.dependencygraph):- tree of analyzed node

        Returns:
          str or None: name of discourse segment or None

        """
        if a_tree.label() == "TOP":
            return "HS"
        else:
            return None

    def _extract_nonadjacent(self, a_seg, a_root_pos):
        """Remove from segment nodes which are not adjacent to root.

        Args:
          a_seg (list): terminals to be modified
          a_root_idx (int): index of the root node

        Return:
          list: non-adjacent words (discourse segment will also be
            modified)

        """
        temp = []
        adjacent = []
        non_adjacent = []
        prev_pos = -2
        root_seen = False
        for cur_pos, cur_node in a_seg.leaves:
            if prev_pos != -2 and prev_pos != cur_pos - 1:
                if root_seen:
                    adjacent += temp
                    root_seen = False
                else:
                    non_adjacent += temp
                del temp[:]
            if cur_pos == a_root_pos or a_root_pos == 0:
                root_seen = True
            if isinstance(cur_node, DiscourseSegment) and cur_node.leaves:
                prev_pos = cur_node.get_end()
            else:
                prev_pos = cur_pos
            temp.append((cur_pos, cur_node))
        if root_seen:
            adjacent += temp
        else:
            non_adjacent += temp
        a_seg.leaves = adjacent
        return non_adjacent

    def _unite_nonadjacent(self, a_word_seg):
        """Add nodes between the discourse segment and its non-projective edges

        Args:
          a_word_seg (list): terminals and segments to be modified

        Returns:
          list: modified word/segment list

        """
        word_seg = []
        right_leaves = []
        cur_node = None
        prev_wseg = nxt_wseg = None
        cur_start = cur_end = wseg_end = -1
        i = 0
        max_i = len(a_word_seg)
        while i < max_i:
            cur_start, cur_node = a_word_seg[i]
            i += 1
            if isinstance(cur_node, DiscourseSegment):
                while word_seg:
                    prev_wseg = word_seg[-1]
                    if isinstance(prev_wseg, DiscourseSegment):
                        wseg_end = word_seg[-1].get_end()
                    else:
                        wseg_end = word_seg[-1][0]
                    if wseg_end > cur_start:
                        cur_node.insort(word_seg.pop())
                    else:
                        break
                cur_end = cur_node.get_end()
                while i < max_i and a_word_seg[i][0] < cur_end:
                    nxt_wseg = a_word_seg[i]
                    if isinstance(nxt_wseg[1], DiscourseSegment):
                        cur_end = max(cur_end, nxt_wseg[1].get_end())
                    right_leaves.append(nxt_wseg)
                    i += 1
                if right_leaves:
                    # right_leaves = self._unite_nonadjacent(right_leaves)
                    # for rl in right_leaves:
                    #     cur_node.insort(rl)
                    cur_node.leaves += right_leaves
                    del right_leaves[:]
                    cur_node.leaves.sort(key=lambda el: el[0])
                    cur_node.leaves[:] = \
                        self._unite_nonadjacent(cur_node.leaves)
            word_seg.append((cur_start, cur_node))
        return word_seg
