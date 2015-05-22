#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""
Module providing discourse segmenter for constituency trees.

Constants:
SUBSTITUTEF - custom weighting function used for token alignment

Methods:
_ispunct - check if word consists only of punctuation characters
_prune_punc - remove tokens representing punctuation from set
_translate_toks - replace tokens and return updated set
tree2tok - create dictionary mapping constituency trees to numbered tokens
read_trees - read file and return a list of constituent dictionaries
read_segments - read file and return a list of segment dictionaries
trees2segs - align trees with corresponding segments

Classes:
BparSegmenter - discourse segmenter for constituency trees

Exceptions:

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Libraries
from align import nw_align
from treeseg import TreeSegmenter, DiscourseSegment, CONSTITUENCY, DEFAULT_SEGMENT
from constituency_tree import Tree, CTree

from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

import argparse
import codecs
import glob
import locale
import numpy as np
import os
import re
import sys

##################################################################
# Constants
N_FOLDS = 10
SUBSTITUTEF = lambda c1, c2: 2 if c1[-1] == c2[-1] else -3
ESCAPE_QUOTE_RE = re.compile(r"\\+([\"'])")
ESCAPE_SLASH_RE = re.compile(r"\\/")

##################################################################
# Methods
locale.setlocale(locale.LC_ALL, '')

def _ispunct(a_word):
    """
    Check if word consists only of punctuation characters

    @param a_word - word to check

    @return True if word consists only of punctuation characters, False otherwise
    """
    return all(c in string.punctuation for c in a_word)

def _prune_punc(a_toks):
    """
    Remove tokens representing punctuation from set

    @param a_toks - tokens to prune

    @return token set with punctuation tokens removed
    """
    return frozenset([tok for tok in a_toks if not _ispunct(tok[-1])])

def _translate_toks(a_toks, a_translation):
    """
    Translate tokens and return translated set

    @param a_toks - tokens to be translated
    @param a_translation - translation dictionary for tokens

    @return translated set of tokens
    """
    if a_translation is None:
        return a_toks
    ret = set()
    for tok in a_toks:
        for t_tok in a_translation[tok]:
            ret.add(t_tok)
    return frozenset(ret)

def tree2tok(a_cls, a_tree, a_start = 0):
    """
    Create dictionary mapping constituency trees to numbered tokens

    @param a_cls - reference to this class
    @param a_tree - tree to analyze
    @param a_start - starting position of the first token

    @return dictionary mapping subtrees to their yields
    """
    rset = set()
    chset = None
    tr2tk = {(a_start, a_tree.label()): (a_tree, rset)}
    i = a_start
    max_ch_pos = -1
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

def read_trees(a_fname, a_one_per_line = False):
    """
    Read file and return a list of constituent dictionaries

    @param a_fname - name of file to be read

    @return list of dictionaries mapping tokens to trees and a list of trees
    """
    ctrees = CTree.parse_file(a_fname, a_encoding = ENCODING, \
                                  a_one_per_line = a_one_per_line)
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

def read_segments(a_fname):
    """
    Read file and return a list of segment dictionaries

    @param a_fname - name of file to be read

    @return dictionary which maps tokens to segments
    """
    segs2toks = {}
    s_c = t_c = 0
    tokens = []
    atoks = []
    new_seg = None
    active_tokens = set()
    active_segments = []
    # read segments
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            # do some clean-up
            active_tokens.clear()
            del atoks[:]
            del active_segments[:]
            tokens = iline.split()
            # establish correspondence between tokens and segments
            for tok in tokens:
                if tok[0] == '(' and len(tok) > 1:
                    active_tokens = set(atoks)
                    del atoks[:]
                    for a_s in active_segments:
                        segs2toks[a_s].update(active_tokens)
                    new_seg = (s_c, tok[1:])
                    active_segments.append(new_seg)
                    segs2toks[new_seg] = set()
                    s_c += 1
                    continue
                elif tok == ')':
                    assert active_segments, "Unbalanced closing parenthesis at line: " + repr(iline)
                    active_tokens = set(atoks)
                    del atoks[:]
                    for a_s in active_segments:
                        segs2toks[a_s].update(active_tokens)
                    active_segments.pop()
                    continue
                else:
                    atoks.append((t_c, tok))
                    t_c += 1
            assert not active_segments, "Unbalanced opening parenthesis at line: " + repr(iline)
    toks2segs = dict()
    segments = segs2toks.keys()
    segments.sort(key = lambda el: el[0])
    for seg in segments:
        toks = frozenset(segs2toks[seg])
        # it can be same tokenset corresponds to multiple segments, in that
        # case we leave the first one that we encounter
        if toks in toks2segs:
            continue
        assert toks not in toks2segs, "Multiple segments correspond to the same tokenset: '" + \
            repr(toks) + "': " + repr(seg) + ", " + repr(toks2segs[toks])
        toks2segs[toks] = seg
    return toks2segs

def trees2segs(a_toks2trees, a_toks2segs):
    """
    Align trees with corresponding segments

    @param a_toks2trees - dictionary mapping tokens to trees
    @param a_toks2segs - dictionary mapping tokens to segments

    @return tree-segment dictionary
    """
    # prune empty trees and their corresponding segments
    tree2seg = {t: None for val in a_toks2trees.values() for t in val}
    # add additional keys to `a_toks2trees` by pruning punctuation marks from
    # existing trees
    pruned_toks = None
    tree_tok_keys = a_toks2trees.keys()
    for tree_toks in tree_tok_keys:
        pruned_toks = _prune_punc(tree_toks)
        if pruned_toks not in a_toks2trees:
            a_toks2trees[pruned_toks] = a_toks2trees[tree_toks]
    # establish a mapping between tree tokens and segment tokens
    tree_toks = list(set([t for t_set in a_toks2trees.keys() for t in t_set]))
    tree_toks.sort(key = lambda el: el[0])
    seg_toks = list(set([t for t_set in a_toks2segs.keys() for t in t_set]))
    seg_toks.sort(key = lambda el: el[0])
    # align tokens if necessary
    seg_t2tree_t = None
    if tree_toks != seg_toks:
        seg_t2tree_t = dict()
        alignment = nw_align(seg_toks, tree_toks, substitute = SUBSTITUTEF, keep_deleted = True)
        for i, tt in enumerate(alignment):
            seg_t2tree_t[seg_toks[i]] = [tree_toks[j] for j in tt]
        # for each segment look if its corresponding token set is matched by
        # any other subtree
        translated_toks = None
    for toks, segs in a_toks2segs.iteritems():
        translated_toks = _translate_toks(toks, seg_t2tree_t)
        key = None
        if translated_toks in a_toks2trees:
            key = translated_toks
        else:
            translated_toks = _prune_punc(translated_toks)
            if translated_toks in a_toks2trees:
                key = translated_toks
        if key:
            for tree in a_toks2trees[key]:
                # if tree2seg[tree] is not None:
                #     continue
                assert tree2seg[tree] is None, "Multiple segments found for tree" + repr(tree) + ": " +\
                    repr(segs[-1]) + "; " + repr(tree2seg[tree])
                tree2seg[tree] = segs[-1]
    return tree2seg

##################################################################
# Class
class BparSegmenter(object):
    """
    Class for perfoming discourse segmentation on constituency trees.

    Constants:
    PIPELINE - default pipeline object used for classification

    Class methods:
    featgen - default feature generation function
    classify - default classification method

    Instance variables:
    model - path to the model that is used in classification
    featgen - pointer to function that is used for feature generation
    classify - pointer to function that is used for classification

    Public instance methods:
    segment - function for doing doiscourse segmentation on BitPar trees
    cv_train - train new model in cross-validation mode and pick the best one
    train - train and store new model
    test - evalute the model on test data

    """

    PIPELINE = Pipeline([('vectorizer', DictVectorizer()),
                         ('var_filter', VarianceThreshold()),
                         ('LinearSVC', LinearSVC(C = 0.3, multi_class = 'crammer_singer'))])
    @classmethod
    def featgen(a_cls, a_tree):
        """
        Generate features for the given BitPar tree.

        @param a_cls - reference to this class
        @param a_tree - BitPar tree for which we should generate features

        @return list of string features
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

    @classmethod
    def classify(a_cls, a_model, a_el, a_default = None):
        """
        Classify given element.

        @param a_cls - reference to this class
        @param a_model - model whch should make predictions
        @param a_el - constituency tree to be classified

        @return assigned class
        """
        prediction = a_model.predict(featgen(a_el))[0]
        return a_default if prediction.lower() == "none" else prediction

    def __init__(self):
        """
        Class constructor.
        """
        self.model = DEFAULT_MODEL
        self.featgen = BparSegmenter.featgen
        self.classify = BparSegmenter.classify

    def segment(self, a_trees):
        """
        Create discourse segments based on the BitPar trees.

        @param a_trees - list of sentence trees to be parsed

        @return iterator over constructed segment trees
        """
        if self.model is None:
            raise RuntimeError("Classification model does not exist.")
        if self.featgen is None:
            raise RuntimeError("Featire generation function does not exist.")
        decfunc = lambda el: self.classify(self.model, el)
        # construct tree segmenter
        tsegmenter = TreeSegmenter(a_decfunc = decfunc, a_type = CONSTITUENCY)
        lines = []
        slines = ""
        seg_idx = 0
        segments = []
        for t in trees:
            tsegmenter.segment(t, segments, decfunc)
            # if classifier failed to create one common segment for
            # the whole tree, create one for it
            if (len(segments) - seg_idx) > 1:
                segments[seg_idx:] = [DiscourseSegment(a_name = DEFAULT_SEGMENT, \
                                                           a_leaves = segments[seg_idx:])]
            seg_idx = len(segments)
        return segments

    def cv_train(self, a_fname2trees, a_fname2featseg, a_out_dir, a_out_sfx):
        """
        Train and store a segmenter model.

        @param a_fname2trees - dictionary mapping file names to sentence trees
        @param a_fname2featseg - dictionary mapping file names to feature-segment pairs
        @param a_out_dir - directory for writing output files
        @param a_out_sfx - suffix which should be appended to output files

        @return \c 0 on success, non-\c 0 otherwise
        """
        fnames = a_fname2featseg.keys()
        n_fnames = len(fnames)
        if n_fnames < self.n_folds:
            print >> sys.stderr, "Insufficient number of samples for cross-validation: {:d}.".format(\
                n_fnames)
            return -1

        processed_fnames = dict()
        folds = KFold(len(fnames), min(len(fnames), N_FOLDS))

        best_i = -1
        best_macro_f1 = float("-inf")
        macro_f1 = 0; macro_F1s = []
        micro_f1 = 0; micro_F1s = []
        pred_segs = []
        pipeline = None
        test_fname = ""
        out_fname = ""
        out_fnames = []
        istart = ilen = 0
        trees = []
        fname2range = {}
        fname2gld_pred = {}
        train_segs = None; test_segs = None
        train_feats = None; test_feats = None
        for i, (train, test) in enumerate(folds):
            print >> sys.stderr, "Fold: {:d}".format(i)
            train_feats = [ftseg[0] for k in train for ftseg in a_fname2featseg[fnames[k]]]
            train_segs = [str(ftseg[1]) for k in train for ftseg in a_fname2featseg[fnames[k]]]

            istart = 0
            for k in test:
                ilen = len(a_fname2featseg[fnames[k]])
                fname2range[fnames[k]] = [istart, istart + ilen]
                istart += ilen
            test_feats = [ftseg[0] for k in test for ftseg in a_fname2featseg[fnames[k]]]
            test_segs = [str(ftseg[1]) for k in test for ftseg in a_fname2featseg[fnames[k]]]
            # train classifier
            pipeline = Pipeline([('vectorizer', DictVectorizer()),
                                 ('var_filter', VarianceThreshold()),
                                 #('feature_selection', SelectKBest(k=5000)),
                                 # ("KDT", KNeighborsClassifier())])
                                 # ('SGD', SGDClassifier(loss="hinge", penalty="l2"))])
                                 # ('to_dense', DenseTransformer()),
                                 # ("RFC", RandomForestClassifier())])
                                 # ('Tree', DecisionTreeClassifier())])
                                 ('LinearSVC', LinearSVC(C = 0.3, multi_class = 'crammer_singer'))])
            pipeline.fit(train_feats, train_segs)
            # obtain new predictions
            pred_segs = pipeline.predict(test_feats)
            # update F1 scores
            _, _, macro_f1, _ = precision_recall_fscore_support(test_segs, pred_segs, average='macro', \
                                                                    pos_label=None)
            _, _, micro_f1, _ = precision_recall_fscore_support(test_segs, pred_segs, average='micro', \
                                                                    pos_label=None)
            macro_F1s.append(macro_f1); micro_F1s.append(micro_f1)
            print >> sys.stderr, "Macro F1: {:.2%}".format(macro_f1)
            print >> sys.stderr, "Micro F1: {:.2%}".format(micro_f1)
            # update maximum macro F-score and store the most successful model
            if macro_f1 > best_macro_f1:
                best_i = i
                best_macro_f1 = macro_f1
                joblib.dump(pipeline, a_model)
            # generate new output files, if necessary
            for k in test:
                test_fname = fnames[k]
                if test_fname in processed_fnames and processed_fnames[test_fname] > macro_f1:
                    continue
                fname2gld_pred[test_fname] = [(test_segs[i], pred_segs[i]) \
                                              for i in xrange(*fname2range[test_fname])]
                processed_fnames[test_fname] = macro_f1
                trees.append(a_fname2trees[test_fname])
                out_fname = os.path.join(a_out_dir, os.path.splitext(test_fname)[0] + a_out_sfx)
                out_fnames.append(out_fname)
            bpar_segmenter_segment(pipeline, featgen, trees, out_fnames)
            del trees[:]; del out_fnames[:]; fname2range.clear()
        print >> sys.stderr, "Average macro F1: {:.2%} +/- {:.2%}".format(np.mean(macro_F1s), \
                                                                              np.std(macro_F1s))
        print >> sys.stderr, "Average micro F1: {:.2%} +/- {:.2%}".format(np.mean(micro_F1s), \
                                                                              np.std(micro_F1s))
        print >> sys.stderr, "Best model obtained in fold {:d}".format(best_i)
        return 0

    def train(self, a_model, a_fname2featseg):
        """
        Train segmenter model

        @param a_model - file to which the best model should be stored
        @param a_fname2featseg - dictionary mapping file names to feature-segment pairs
        @param a_fname2trees - dictionary mapping file names to sentence trees

        @return \c 0 on success, non-\c 0 otherwise
        """
        ret = 0
        train_feats = [ftseg[0] for fname in a_fname2featseg for ftseg in a_fname2featseg[fname]]
        train_segs = [str(ftseg[1]) for fname in a_fname2featseg for ftseg in a_fname2featseg[fname]]
        # train classifier
        pipeline = Pipeline([('vectorizer', DictVectorizer()),
                             ('var_filter', VarianceThreshold()),
                             #('feature_selection', SelectKBest(k=5000)),
                             ('LinearSVC', LinearSVC(class_weight='auto'))])
        pipeline.fit(train_feats, train_segs)
        joblib.dump(pipeline, a_model)
        return ret

    def test(a_model, a_out_dir, a_out_sfx, a_bpf2trees):
        """
        Train and store a segmenter model.

        @param a_model - file to which the best model should be stored
        @param a_out_dir - directory for writing output files
        @param a_out_sfx - suffix which should be appended to output files
        @param a_fname2trees - dictionary mapping file names to sentence trees

        @return \c 0 on success, non-\c 0 otherwise
        """
        ret = 0
        pipeline = joblib.load(a_model)
        for iname, itrees in a_bpf2trees.iteritems():
            trees.append(itrees)
            out_fnames.append(os.path.join(a_out_dir, os.path.splitext(fname)[0] + a_out_sfx))
        bpar_segmenter_segment(pipeline, featgen, trees, out_fnames)
        return ret
