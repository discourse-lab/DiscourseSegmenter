#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Created on 03.01.2015

@author: Andreas Peldszus

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsegmenter.common import NONE, prune_punc, score_substitute, \
    translate_toks

from dsegmenter.mateseg.dependency_graph import HEAD, WORD, REL, TAG, ADDRESS
from dsegmenter.mateseg.segmentation_tree import generate_subtrees_from_forest
from dsegmenter.treeseg import (TreeSegmenter, DiscourseSegment, DEPENDENCY,
                                DEFAULT_SEGMENT)
from dsegmenter.treeseg.treesegmenter import NO_MATCH_STRING
from dsegmenter.treeseg.constants import GREEDY
from dsegmenter.bparseg.align import nw_align

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

import os
import sys

##################################################################
# Variables and Constants
number_of_folds = 10
punct_tags = ['$.', '$,']
feature_not_found = '[NONE]'
DEFAULT_TEXT_ID = 'text'
PREDICTION = 'prediction'


##################################################################
# Methods
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


def gen_features_for_segment(dep_graph, trg_adr):
    """ ugly feature extraction code  ;) """

    nodes = list(dep_graph.subgraphs(exclude_root=False))
    nl = {node[ADDRESS]: node for node in nodes}
    assert len(nodes) == len(nl)

    if trg_adr >= len(nl):
        return {}
    seg_adr_span = dep_graph.address_span(trg_adr)

    # get relation, this word and pos
    rel = nl[trg_adr][REL] if REL in nl[trg_adr] else feature_not_found
    this_word = nl[trg_adr][WORD][1] if WORD in nl[trg_adr] and \
        nl[trg_adr][WORD] is not None else feature_not_found
    this_pos = nl[trg_adr][TAG] if TAG in nl[trg_adr] else feature_not_found

    # get head word and pos
    head_adr = nl[trg_adr][HEAD] if HEAD in nl[trg_adr] else None
    if head_adr is not None:
        head_word = nl[head_adr][WORD][1] if WORD in nl[head_adr] and \
            nl[head_adr][WORD] is not None else feature_not_found
        head_pos = nl[head_adr][TAG] if TAG in nl[head_adr] else \
            feature_not_found
    else:
        head_word = head_pos = feature_not_found

    # get first and last word from segment
    first_adr = seg_adr_span[0]
    first_word = nl[first_adr][WORD][1] if nl[first_adr][WORD] is not None \
        else feature_not_found
    last_adr = seg_adr_span[-1]
    last_word = nl[last_adr][WORD][1] if nl[last_adr][WORD] is not None \
        else feature_not_found

    # get words left and right from segment
    left_adr = seg_adr_span[0] - 1
    left_word = nl[left_adr][WORD][1] if len(nl) > left_adr > 0 and \
        WORD in nl[left_adr] and nl[left_adr][WORD] is not None else \
        feature_not_found
    right_adr = seg_adr_span[-1] + 1
    right_word = nl[right_adr][WORD][1] if len(nl) > right_adr > 0 and \
        WORD in nl[right_adr] and nl[right_adr][WORD] is not None else \
        feature_not_found

    # get segment length
    length_abs = len(seg_adr_span)

    # get number of punctuation in segment
    punct_count = sum([1 if adr in nl and TAG in nl[adr] and
                       nl[adr][TAG] in punct_tags else 0 for adr in
                       seg_adr_span])

    # resulting feature dictionary
    r = {
        'rel': rel,
        'head_word': head_word,
        'head_pos': head_pos,
        'this_word': this_word,
        'this_pos': this_pos,
        'rel+head_pos+this_pos': '+'.join([rel, head_pos, this_pos]),
        'rel+head_pos': '+'.join([rel, head_pos]),
        'rel+this_pos': '+'.join([rel, this_pos]),
        'head_pos+this_pos': '+'.join([head_pos, this_pos]),
        'first_word': first_word,
        'last_word': last_word,
        'left_word': left_word,
        'right_word': right_word,
        'length_abs': length_abs,
        'punct_count': punct_count,
    }

    # simply add all words for this segment
    for adr in seg_adr_span:
        if nl[adr][WORD] is not None:
            word = nl[adr][WORD][1]
            r['word_%s' % word] = 1
    return r


def word_access(x):
    if x is None:
        return ''
    else:
        return x[1]


def substitution_costs(c1, c2):
    """defines the costs of substitutions for the alignment"""
    if c1[-1] == c2[-1]:
        return 2
    else:
        return -3


def chained(iterable):
    """flattens a single embed iterable"""
    return list(elm for sublist in iterable for elm in sublist)


def get_observations(seg_trees, dep_trees):
    if seg_trees is None:
        return get_testing_observations(dep_trees)
    else:
        return get_training_observations(seg_trees, dep_trees)


def get_testing_observations(dep_trees):
    for sentence_index, dep_tree in enumerate(dep_trees):
        for node in dep_tree.subgraphs(exclude_root=True):
            yield (sentence_index, node[ADDRESS], dep_tree, None)


def get_training_observations(seg_trees, dep_trees):
    # pregenerate all dependency subgraphs
    items = []
    for sentence_index, dep_tree in enumerate(dep_trees):
        for node in dep_tree.subgraphs(exclude_root=True):
            tokset = set(dep_tree.token_span(node[ADDRESS]))
            items.append((sentence_index, node[ADDRESS], dep_tree, tokset))

    # match tokenization first
    seg_tokens = chained([tree.leaves() for tree in seg_trees])
    dep_tokens = chained([dg.words() for dg in dep_trees])
    unequal_tokenizations = False
    if seg_tokens != dep_tokens:
        unequal_tokenizations = True
        aligned = nw_align(dep_tokens, seg_tokens,
                           substitute=substitution_costs, keep_deleted=True)
        # make a translation
        seg_to_dep_tok = {}
        for dep_index, seg_index_list in enumerate(aligned):
            for seg_index in seg_index_list:
                seg_to_dep_tok[seg_tokens[seg_index]] = dep_tokens[dep_index]

    # match every dep_tree subgraphs with all seg_tree non-terminals
    for sentence_index, address, dep_tree, tokset in items:
        found_match = False
        for seg_sub_tree in generate_subtrees_from_forest(seg_trees):
            node = seg_sub_tree.label()
            if node is None or node == "":
                print("Warning: Empty node.", sentence_index,
                      file=sys.stderr)
            if unequal_tokenizations:
                seg_leaves = set([seg_to_dep_tok[leaf]
                                 for leaf in seg_sub_tree.leaves()])
            else:
                seg_leaves = set(seg_sub_tree.leaves())
            if seg_leaves == tokset:
                found_match = True
                yield (sentence_index, address, dep_tree, node)
                break
        if not found_match:
            yield (sentence_index, address, dep_tree, NO_MATCH_STRING)


def _cnt_stat(a_gold_segs, a_pred_segs):
    """Estimate the number of true pos, false pos, and false neg.

    Args:
      a_gold_segs (iterable): gold segments
      a_pred_segs (iterable): predicted segments

    Returns:
     tuple: true positives, false positives, and false negatives

    """
    tp = fp = fn = 0
    for gs, ps in zip(a_gold_segs, a_pred_segs):
        gs = gs.lower()
        ps = ps.lower()
        if gs == "none":
            if ps != "none":
                fp += 1
        elif gs == ps:
            tp += 1
        else:
            fn += 1
    return tp, fp, fn


def decision_function(node, tree):
    """decision function for the tree segmenter"""
    assert PREDICTION in node, "No prediction for node {}".format(node)
    pred = node[PREDICTION]
    # pred = NO_MATCH_STRING
    if (pred == NO_MATCH_STRING or pred == NONE) and HEAD in node \
       and node[HEAD] == 0:
        # The classifier did not recognize sentence top as a segment, so we
        # enforce a labelling with the default segment type.
        return DEFAULT_SEGMENT
    else:
        return pred


##################################################################
# Class
class MateSegmenter(object):
    """Class for perfoming discourse segmentation on constituency trees.

    """

    #: classifier object: default classification method
    DEFAULT_CLASSIFIER = LinearSVC(multi_class="ovr",
                                   class_weight="balanced")

    #: path  to default model to use in classification
    DEFAULT_MODEL = os.path.join(os.path.dirname(__file__),
                                 "data", "mate.model")

    #: default pipeline object used for classification
    DEFAULT_PIPELINE = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('var_filter', VarianceThreshold()),
        ('classifier', DEFAULT_CLASSIFIER)])

    def __init__(self, featgen=gen_features_for_segment, model=DEFAULT_MODEL):
        """Class constructor.

        """
        self.featgen = featgen
        self.model = None
        self._update_model(model)
        self._segmenter = TreeSegmenter(a_type=DEPENDENCY)

    def extract_features_from_text(self, dep_forest, seg_forest=None):
        """Extract features from dependency trees.

        Args:
          dep_forrest (list): list of sentence trees to be parsed
          dep_forrest (list or None): list of discourse segments

        Returns:
          2-tuple[list, list]: list of features and list of labels

        """
        features = []
        labels = []
        observations = get_observations(seg_forest, dep_forest)
        for sentence_index, address, dep_tree, class_ in sorted(observations):
            features.append(self.featgen(dep_tree, address))
            labels.append(class_)
        return features, labels

    def segment(self, a_trees):
        """Create discourse segments based on the Mate trees.

        Args:
          a_trees (list): list of sentence trees to be parsed

        Returns:
          list: constructed segment trees

        """
        segments = []
        features = predictions = None
        for itree in a_trees:
            features, _ = self.extract_features_from_text([itree])
            predictions = self._predict(features)
            segments.append(self._segment_sentence(
                predictions, itree)[0][1])
        return (segments,)

    def segment_text(self, dep_forest):
        """Segment all sentences of a text.

        Args:
          dep_forrest (list[dsegmenter.mateseg.dependency_graph]): list
            of sentence trees to be parsed

        Returns:
          list: constructed segment trees

        """
        features = self.extract_features_from_text(dep_forest)
        predictions = self._predict(features)
        return self._segment_text(predictions, dep_forest)

    def _segment_text(self, predictions, parses):
        all_segments = []
        for sentence, dep_graph in enumerate(parses):
            # slice prediction vector
            sentence_length = dep_graph.length()
            sentence_predictions = predictions[:sentence_length]
            predictions = predictions[sentence_length:]
            # segment
            segments = self._segment_sentence(sentence_predictions, dep_graph)
            segment = segments[0][1]
            all_segments.append((sentence, segment))
        return DiscourseSegment(a_name=DEFAULT_SEGMENT, a_leaves=all_segments)

    def _segment_sentence(self, sentence_predictions, dep_graph):
        if dep_graph.is_valid_parse_tree():
            # remove prediction annotations (just to be sure)
            dep_graph.deannotate(PREDICTION)
            # annotate dep_graph with sentence predictions
            dep_graph.annotate(sentence_predictions, PREDICTION)
            # call tree_segmenter
            segments = self._segmenter.segment(
                dep_graph, a_predict=decision_function,
                a_word_access=word_access, a_strategy=GREEDY,
                a_root_idx=dep_graph.root[ADDRESS])
            if len(segments) > 1:
                segments = [(0, DiscourseSegment(a_name=DEFAULT_SEGMENT,
                                                 a_leaves=segments))]
        else:
            # make a simple sentence segment for invalid parse trees
            leaves = [(i, word) for i, (_, word) in
                      enumerate(dep_graph.words(), 1)]
            dseg = DiscourseSegment(a_name=DEFAULT_SEGMENT, a_leaves=leaves)
            segments = [(0, dseg)]
        return segments

    def train(self, trees, segments, path=None):
        """Train segmenter model.

        Args:
          a_trees (list): BitPar trees
          a_segs (list): discourse segments
          a_path (str): path to file in which the trained model should be
                        stored

        Returns:
          void:

        """
        features = [self.featgen(t, n) for t, n in trees]
        segments = [str(s) for s in segments]
        self._train(features, segments)
        if path is not None:
            joblib.dump(self.model, path, compress=1, cache_size=1e9)

    def _train(self, features, labels):
        self.model = MateSegmenter.DEFAULT_PIPELINE
        self.model.fit(features, labels)

    def test(self, trees, segments):
        """Estimate performance of segmenter model.

        Args:
          a_trees (list): BitPar trees
          a_segments (list): corresponding gold segments for trees

        Returns:
          2-tuple: macro and micro-averaged F-scores

        """
        predictions = [self.model.predict(self.featgen(t, n))
                       for t, n in trees]
        segments = [str(s) for s in segments]
        return self._score(segments, predictions)

    def _predict(self, features):
        return [None if p == NONE else p
                for p in self.model.predict(features)]

    def _score(self, labels, predicted_labels):
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average='macro', warn_for=())
        _, _, micro_f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average='micro', warn_for=())
        return macro_f1, micro_f1

    def _update_model(self, model):
        if model is None:
            self.model = MateSegmenter.DEFAULT_PIPELINE
        elif isinstance(model, basestring):
            if not os.path.isfile(model) or not os.access(model, os.R_OK):
                raise RuntimeError("Can't load model from file {:s}".format(
                    model))
            self.model = joblib.load(model)
        else:
            self.model = model
