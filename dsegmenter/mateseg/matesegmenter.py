#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
Created on 03.01.2015

@author: Andreas Peldszus
'''

##################################################################
# Imports
from .dependency_graph import HEAD, WORD, REL, TAG, ADDRESS
from .segmentation_tree import generate_subtrees_from_forest
from ..treeseg import (TreeSegmenter, DiscourseSegment, DEPENDENCY,
                       DEFAULT_SEGMENT)
from ..treeseg.treesegmenter import NO_MATCH_STRING
from ..treeseg.constants import GREEDY
from ..bparseg.align import nw_align

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

import numpy as np
import os

##################################################################
# Variables and Constants
number_of_folds = 10
punct_tags = ['$.', '$,']
feature_not_found = '[NONE]'
DEFAULT_TEXT_ID = 'text'
PREDICTION = 'prediction'


##################################################################
# Methods
def gen_features_for_segment(dep_graph, trg_adr):
    ''' ugly feature extraction code  ;) '''

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
    '''defines the costs of substitutions for the alignment'''
    if c1[-1] == c2[-1]:
        return 2
    else:
        return -3


def chained(iterable):
    '''flattens a single embed iterable'''
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
                print "Warning: Empty node.", sentence_index
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
    '''decision function for the tree segmenter'''
    assert PREDICTION in node, "No prediction for node {}".format(node)
    pred = node[PREDICTION]
    # pred = NO_MATCH_STRING
    if pred == NO_MATCH_STRING and HEAD in node and node[HEAD] == 0:
        # The classifier did not recognize sentence top as a segment, so we
        # enforce a labelling with the default segment type.
        return DEFAULT_SEGMENT
    else:
        return pred


class MateSegmenter(object):
    """Class for perfoming discourse segmentation on constituency trees.

    """

    #: classifier object: default classification method
    DEFAULT_CLASSIFIER = LinearSVC(multi_class='ovr', class_weight='auto')

    #:str: path  to default model to use in classification
    DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "data", "mate.model")

    #:pipeline object: default pipeline object used for classification
    DEFAULT_PIPELINE = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('var_filter', VarianceThreshold()),
        ('classifier', DEFAULT_CLASSIFIER)])

    def __init__(self, featgen=gen_features_for_segment, model=DEFAULT_MODEL):
        """Class constructor.
        """
        self.featgen = featgen
        self.pipeline = None
        self._update_model(model)

    def extract_features_from_corpus(self, dep_corpus, seg_corpus=None):
        all_features = []
        all_labels = []
        for text in sorted(dep_corpus.keys()):
            seg_forest = seg_corpus.get(text, None)
            features, labels = self.extract_features_from_text(
                dep_corpus[text], seg_forest=seg_forest)
            all_features.extend(features)
            all_labels.extend(labels)
        return all_features, all_labels

    def extract_features_from_text(self, dep_forest, seg_forest=None):
        features = []
        labels = []
        observations = get_observations(seg_forest, dep_forest)
        for sentence_index, address, dep_tree, class_ in sorted(observations):
            features.append(self.featgen(dep_tree, address))
            labels.append(class_)
        return features, labels

    def segment(self, dep_corpus, out_folder):
        for text, trees in dep_corpus.iteritems():
            print text
            discourse_tree = self.segment_text(trees)
            with open(out_folder + '/' + text + '.tree', 'w') as fout:
                fout.write(str(discourse_tree))

    def segment_text(self, dep_forest):
        features, _ = self.extract_features_from_text(dep_forest)
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
        return DiscourseSegment(a_name='TEXT', a_leaves=all_segments)

    def _segment_sentence(self, sentence_predictions, dep_graph):
        if dep_graph.is_valid_parse_tree():
            # remove prediction annotations (just to be sure)
            dep_graph.deannotate(PREDICTION)
            # annotate dep_graph with sentence predictions
            dep_graph.annotate(sentence_predictions, PREDICTION)
            # call tree_segmenter
            segmenter = TreeSegmenter(a_type=DEPENDENCY)
            segments = segmenter.segment(
                dep_graph, a_predict=decision_function,
                a_word_access=word_access, a_strategy=GREEDY,
                a_root_idx=dep_graph.root[ADDRESS])
        else:
            # make a simple sentence segment for invalid parse trees
            leaves = [(i, word) for i, (_, word) in
                      enumerate(dep_graph.words(), 1)]
            dseg = DiscourseSegment(a_name=DEFAULT_SEGMENT, a_leaves=leaves)
            segments = [(0, dseg)]
        return segments

    def train(self, seg_corpus, dep_corpus, path=None):
        assert seg_corpus.keys() == dep_corpus.keys()
        features, labels = self.extract_features_from_corpus(
            dep_corpus, seg_corpus=seg_corpus)
        self._train(features, labels)
        if path is not None:
            joblib.dump(self.pipeline, path, compress=1, cache_size=1e9)

    def _train(self, features, labels):
        self.pipeline = MateSegmenter.DEFAULT_PIPELINE
        self.pipeline.fit(features, labels)

    def test(self, seg_corpus, dep_corpus):
        assert seg_corpus.keys() == dep_corpus.keys()
        features, labels = self.extract_features_from_corpus(
            dep_corpus, seg_corpus=seg_corpus)
        predicted_labels = self._predict(features)
        return self._score(labels, predicted_labels)

    def _predict(self, features):
        return self.pipeline.predict(features)

    def _score(self, labels, predicted_labels):
        _, _, macro_f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average='macro', warn_for=())
        _, _, micro_f1, _ = precision_recall_fscore_support(
            labels, predicted_labels, average='micro', warn_for=())
        return macro_f1, micro_f1

    def cross_validate(self, seg_corpus, dep_corpus, out_folder=None):
        assert seg_corpus.keys() == dep_corpus.keys()
        texts = np.array(sorted(seg_corpus.keys()))
        folds = KFold(len(texts), number_of_folds)

        # extract features for all texts
        all_features = {}
        all_labels = {}
        for text in texts:
            features, labels = self.extract_features_from_text(
                dep_corpus[text], seg_forest=seg_corpus[text])
            all_features[text] = features
            all_labels[text] = labels

        # do the cross-validation
        macro_F1s = []
        micro_F1s = []
        tp = fp = fn = tp_i = fp_i = fn_i = 0
        for i, (train, test) in enumerate(folds):
            print "# FOLD", i
            # train
            train_texts = texts[train]
            train_features = chained([all_features[text] for text in
                                      train_texts])
            train_labels = chained([all_labels[text] for text in train_texts])
            print "  training on %d items..." % len(train_labels)
            self._train(train_features, train_labels)
            print "  extracted %d features using the dict vectorizer." % \
                len(self.pipeline.named_steps[
                    'vectorizer'].get_feature_names())
            # test (predicting textwise)
            test_labels = []
            pred_labels = []
            for text in texts[test]:
                features = all_features[text]
                labels = all_labels[text]
                predictions = self._predict(features)
                test_labels.extend(labels)
                pred_labels.extend(predictions)
                if out_folder is not None:
                    discourse_tree = self._segment_text(predictions,
                                                        dep_corpus[text])
                    with open(out_folder + '/' + text + '.tree', 'w') as fout:
                        fout.write(str(discourse_tree))
            macro_f1, micro_f1 = self._score(test_labels, pred_labels)
            macro_F1s.append(macro_f1)
            micro_F1s.append(micro_f1)
            tp_i, fp_i, fn_i = _cnt_stat(test_labels, pred_labels)
            tp += tp_i
            fp += fp_i
            fn += fn_i

        print "# Average Macro F1 = %3.1f +- %3.2f" % \
            (100 * np.mean(macro_F1s), 100 * np.std(macro_F1s))
        print "# Average Micro F1 = %3.1f +- %3.2f" % \
            (100 * np.mean(micro_F1s), 100 * np.std(micro_F1s))
        if tp or fp or fn:
            print "# F1_{tp,fp} %.2f" % (2. * tp / (2. * tp + fp + fn) * 100)
        else:
            print "# F1_{tp,fp} 0. %"

    def _update_model(self, model):
        if model is None:
            self.pipeline = MateSegmenter.DEFAULT_PIPELINE
        elif isinstance(model, str):
            if not os.path.isfile(model) or not os.access(model, os.R_OK):
                raise RuntimeError("Can't load model from file {:s}".format(
                    model))
            self.pipeline = joblib.load(model)
        else:
            self.pipeline = model
