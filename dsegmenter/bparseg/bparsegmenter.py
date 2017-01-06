#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Module providing discourse segmenter for constituency trees.

Attributes:
  tree2tok (method): create dictionary mapping constituency trees to numbered
    tokens
  read_trees (method): read file and return a list of constituent dictionaries
  trees2segs (method): align trees with corresponding segments
  featgen (method): default feature generation function
  classify (method): default classification method


.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Libraries
from __future__ import absolute_import, unicode_literals

from dsegmenter.common import NONE, prune_punc, score_substitute, \
    translate_toks

from dsegmenter.bparseg.align import nw_align
from dsegmenter.bparseg.constituency_tree import Tree, CTree
from dsegmenter.treeseg import TreeSegmenter, DiscourseSegment, \
    CONSTITUENCY, DEFAULT_SEGMENT

from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import locale
import os
import re

##################################################################
# Constants
locale.setlocale(locale.LC_ALL, "")
N_FOLDS = 10
ESCAPE_QUOTE_RE = re.compile(r"\\+([\"'])")
ESCAPE_SLASH_RE = re.compile(r"\\/")


##################################################################
# Methods
def tree2tok(a_tree, a_start=0):
    """Create dictionary mapping constituency trees to numbered tokens.

    Args:
      a_tree (constituency_tree.Tree): tree to analyze
      a_start (int): starting position of the first token

    Returns:
      (dict) mapping from subtrees to their yields

    """
    rset = set()
    chset = None
    tr2tk = {(a_start, a_tree.label()): (a_tree, rset)}
    i = a_start
    for child in a_tree:
        if isinstance(child, Tree):
            tr2tk.update(tree2tok(child, i))
            chset = tr2tk[(i, child.label())][-1]
            i += len(chset)
            rset.update(chset)
        else:
            rset.add((i, child))
            i += 1
    return tr2tk


def read_tok_trees(a_lines, a_one_per_line=False):
    """Read file and return a list of constituent dictionaries.

    Args:
      a_lines (list[str]): decoded lines of the input file
      a_one_per_line (bool): boolean flag indicating whether each
        tree is stored on a separate line

    Returns:
      2-tuple: list of dictionaries mapping tokens to trees and a list of trees

    """
    ctrees = CTree.parse_lines(a_lines, a_one_per_line=a_one_per_line)
    # generate dictionaries mapping trees' yields to trees
    t_cnt = 0
    t2t = None
    trees2toks = dict()
    for ctree in ctrees:
        t2t = tree2tok(ctree, t_cnt)
        trees2toks.update(t2t)
        t_cnt += len(t2t[(t_cnt, ctree.label())][-1])

    toks2trees = dict()
    for ((tree_c, tree_lbl), (tree, toks)) in trees2toks.iteritems():
        toks = frozenset(toks)
        if toks in toks2trees:
            toks2trees[toks].append(tree)
        else:
            toks2trees[toks] = [tree]
    return toks2trees, ctrees


def read_trees(a_lines, a_one_per_line=False):
    """Read file and return a list of constituent dictionaries.

    Args:
      a_lines (list[str]): decoded lines of the input file
      a_one_per_line (bool): boolean flag indicating whether each
        tree is stored on a separate line

    Yields:
      CTree: input tree

    """
    for ctree in CTree.parse_lines(a_lines, a_one_per_line=a_one_per_line):
        yield ctree


def trees2segs(a_toks2trees, a_toks2segs):
    """Align trees with corresponding segments.

    Args:
      a_toks2trees (dict): mapping from tokens to trees
      a_toks2segs (dict): mapping from tokens to segments

    Returns:
      dict: mapping from trees to segments

    """
    # prune empty trees and their corresponding segments
    tree2seg = {t: None
                for val in a_toks2trees.itervalues()
                for t in val}
    # add additional keys to `a_toks2trees` by pruning punctuation marks from
    # existing trees
    pruned_toks = None
    tree_tok_keys = a_toks2trees.keys()
    for tree_toks in tree_tok_keys:
        pruned_toks = prune_punc(tree_toks)
        if pruned_toks not in a_toks2trees:
            a_toks2trees[pruned_toks] = a_toks2trees[tree_toks]
    # establish a mapping between tree tokens and segment tokens
    tree_toks = list(set([t
                          for t_set in a_toks2trees.iterkeys()
                          for t in t_set]))
    tree_toks.sort(key=lambda el: el[0])
    seg_toks = list(set([t
                         for t_set in a_toks2segs.iterkeys()
                         for t in t_set]))
    seg_toks.sort(key=lambda el: el[0])
    # align tokens if necessary
    seg_t2tree_t = None
    if tree_toks != seg_toks:
        seg_t2tree_t = dict()
        alignment = nw_align(seg_toks, tree_toks,
                             substitute=score_substitute,
                             keep_deleted=True)
        for i, tt in enumerate(alignment):
            seg_t2tree_t[seg_toks[i]] = [tree_toks[j] for j in tt]
        # for each segment look if its corresponding token set is matched by
        # any other subtree
        translated_toks = None
    for toks, segs in a_toks2segs.iteritems():
        translated_toks = translate_toks(toks, seg_t2tree_t)
        key = None
        if translated_toks in a_toks2trees:
            key = translated_toks
        else:
            translated_toks = prune_punc(translated_toks)
            if translated_toks in a_toks2trees:
                key = translated_toks
        if key:
            for tree in a_toks2trees[key]:
                # if tree2seg[tree] is not None:
                #     continue
                assert tree2seg[tree] is None, \
                    "Multiple segments found for tree" + repr(tree) + ": " + \
                    repr(segs[-1]) + "; " + repr(tree2seg[tree])
                tree2seg[tree] = segs[-1]
    return tree2seg


def featgen(a_tree):
    """Generate features for the given BitPar tree.

    Args:
     a_tree (dsegmenter.bparseg.constituency_tree.CTree): BitPar tree
      for which we should generate features

    Returns:
      list: string features

    """
    assert a_tree.leaves(), "Tree does not contain leaves."
    # add unigram features
    ret = {u"tok_{:s}".format(token.lower()): 1 for token in a_tree.leaves()}
    # add very first and very last tokens of the tree
    ret[u"tokFirst_{:s}".format(a_tree.leaves()[0].lower())] = 1
    ret[u"tokLast_{:s}".format(a_tree.leaves()[-1].lower())] = 1
    sublabels = [st.label() for st in a_tree.subtrees()]
    if sublabels:
        ret[u"lblFirst_{:s}".format(sublabels[0].lower())] = 1
        ret[u"lblLast_{:s}".format(sublabels[-1].lower())] = 1
    # add tree label
    ret[u"lbl_{:s}".format(a_tree.label())] = 1
    # add label of the parent tree
    ret[u"prntLbl_{:s}".format(a_tree.prnt_label())] = 1
    # add first and last word of the parent tree
    if a_tree.parent():
        prnt_tree = a_tree.parent()
        t_idx = a_tree.parent_index()
        ret[u"treeIdx"] = t_idx
        if t_idx > 0:
            prev_tree = prnt_tree[t_idx - 1]
            ret[u"prevLbl_{:s}".format(prev_tree.label())] = 1
            ret[u"prevTokFrst_{:s}".format(prev_tree.leaves()[0].lower())] = 1
            ret[u"prevTokLst_{:s}".format(prev_tree.leaves()[-1].lower())] = 1
        if t_idx + 1 < len(prnt_tree):
            nxt_tree = prnt_tree[t_idx + 1]
            ret[u"nxtLbl_{:s}".format(nxt_tree.label())] = 1
            ret[u"pxtTokFrst_{:s}".format(nxt_tree.leaves()[0].lower())] = 1
            ret[u"pxtTokLst_{:s}".format(nxt_tree.leaves()[-1].lower())] = 1
    # add tree height
    ret["height"] = a_tree.height()
    # add label of the parent tree
    return ret


def classify(a_classifier, a_featgen, a_el, a_default=None):
    """Classify given element.

    Args:
      a_classifier - model which should make predictions
      a_featgen - feature generation function
      a_el - constituency tree to be classified
      a_default - default element that should be returned if el does
                       not yield segment

    Returns:
      str: assigned class

    """
    prediction = a_classifier.predict(a_featgen(a_el))[0]
    return a_default if prediction is None or \
        prediction == NONE else prediction


##################################################################
# Class
class BparSegmenter(object):
    """Class for perfoming discourse segmentation on constituency trees.

    """

    #:classifier object: default classification method
    DEFAULT_CLASSIFIER = LinearSVC(C=0.3, multi_class='crammer_singer')

    #:str: path  to default model to use in classification
    DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "data",
                                 "bpar.model")

    #:pipeline object: default pipeline object used for classification
    DEFAULT_PIPELINE = Pipeline([('vectorizer', DictVectorizer()),
                                 ('var_filter', VarianceThreshold()),
                                 ('LinearSVC', DEFAULT_CLASSIFIER)])

    def __init__(self, a_featgen=featgen, a_classify=classify,
                 a_model=DEFAULT_MODEL):
        """Class constructor.

        Args:
          a_featgen (method): function to be used for feature generation
          a_classify (method): pointer to 2-arg function which predicts segment
                            class for BitPar tree based on the model and
                            features generated for that tree
          a_model (str): path to a pre-trained model (previously dumped by
                         joblib) or valid classification object or None

        """
        self.featgen = a_featgen
        self.classify = a_classify
        self._update_segmenter(a_model)

    def segment(self, a_trees):
        """Create discourse segments based on the BitPar trees.

        Args:
          a_trees (list): list of sentence trees to be parsed

        Returns:
          list: constructed segment trees

        """
        seg_idx = 0
        segments = []
        isegment = None
        if self.model is None:
            return [DiscourseSegment(a_name=DEFAULT_SEGMENT, a_leaves=t.leaves)
                    for t in a_trees]
        for t in a_trees:
            self._segmenter.segment(t, segments)
            # if classifier failed to create one common segment for
            # the whole tree, create one for it
            if (len(segments) - seg_idx) > 1 or \
                    (len(segments) and not isinstance(segments[-1][-1],
                                                      DiscourseSegment)):
                isegment = DiscourseSegment(a_name=DEFAULT_SEGMENT,
                                            a_leaves=segments[seg_idx:])
                segments[seg_idx:] = [(isegment.leaves[0][0], isegment)]
            seg_idx = len(segments)
        return segments

    def train(self, a_trees, a_segs, a_path):
        """Train segmenter model.

        Args:
          a_trees (list): BitPar trees
          a_segs (list): discourse segments
          a_path (str): path to file in which the trained model should be
                        stored

        Returns:
          void:

        """
        # drop current model
        self._update_segmenter(self.DEFAULT_PIPELINE)
        # generate features
        feats = [self.featgen(t) for t in a_trees]
        a_segs = [str(s) for s in a_segs]
        # train classifier
        self._train(feats, a_segs, self.model)
        # store the model to file
        joblib.dump(self.model, a_path)

    def test(self, a_trees, a_segments):
        """Estimate performance of segmenter model.

        Args:
          a_trees (list): BitPar trees
          a_segments (list): corresponding gold segments for trees

        Returns:
          2-tuple: macro and micro-averaged F-scores

        """
        if self.model is None:
            return (0, 0)
        segments = [self.model.predict(self.featgen(itree))[0]
                    for itree in a_trees]
        a_segments = [str(s) for s in a_segments]
        _, _, macro_f1, _ = precision_recall_fscore_support(a_segments,
                                                            segments,
                                                            average='macro',
                                                            warn_for=())
        _, _, micro_f1, _ = precision_recall_fscore_support(a_segments,
                                                            segments,
                                                            average='micro',
                                                            warn_for=())
        return (macro_f1, micro_f1)

    def _train(self, a_feats, a_segs, a_model):
        """Train segmenter model.

        Args:
          a_feats (list): BitPar featuress
          a_segs (list): discourse segments
          a_model: model object whose parameters should be fit

        Returns:
          void:

        """
        # train classifier
        a_model.fit(a_feats, a_segs)
        self._update_segmenter(a_model)

    def _update_segmenter(self, a_model):
        """Update model, decision function, and internal segmenter.

        Args:
          a_model: model used by classifier

        Returns:
          void:

        """
        if a_model is None:
            self.model = a_model
            self.decfunc = lambda el: None
            self._segmenter = TreeSegmenter(a_decfunc=self.decfunc,
                                            a_type=CONSTITUENCY)
            return
        elif isinstance(a_model, str):
            if not os.path.isfile(a_model) or not os.access(a_model, os.R_OK):
                raise RuntimeError("Can't create model from"
                                   " file {:s}".format(a_model))
            self.model = joblib.load(a_model)
        else:
            self.model = a_model
        self.decfunc = lambda el: self.classify(self.model, self.featgen, el)
        self._segmenter = TreeSegmenter(a_decfunc=self.decfunc,
                                        a_type=CONSTITUENCY)
