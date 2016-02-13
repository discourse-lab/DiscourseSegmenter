#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""

Helper functions for several metrics to calculate the agreement between
two segmentation trees.

@author = Andreas Peldszus
@mail = <peldszus at uni dash potsdam dot de>
@version = 0.1.0

"""

##################################################################
# Imports
import os
import math
import subprocess
from itertools import chain
from collections import Counter

import segeval
import parseval

from .segmentation import get_untyped_masses, get_typed_nonoverlapping_spans


##################################################################
# Methods
def avg(values):
    return sum(values) / len(values)


def sigma(values):
    average = avg(values)
    return math.sqrt(
        sum([(v - average) ** 2 for v in values]) / (len(values) - 1))


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    else:
        return 2.0 * (precision * recall) / (precision + recall)


def metric_accuracy(ground_truth, prediction):
    match = [1 if g == p else 0
             for g, p in zip(ground_truth, prediction)]
    if len(match) > 0:
        return float(sum(match)) / len(match)
    else:
        return float(0)


def metric_kappa(ground_truth, prediction):
    """Calculate Cohen's Kappa

    (Example from Artstein Poesio 2008, p. 568)
    >>> exa = 46 * [(0, 0)] + 6 * [(1, 0)] + 32 * [(1, 1)] + 6 * [(1, 2)] + 10
        * [(2, 2)]

    >>> gold, pred = zip(*exa)
    >>> '%.3f, %.3f, %.3f' % metric_kappa(gold, pred)
    '0.801, 0.396, 0.880'

    """
    assert len(ground_truth) == len(prediction)
    items_count = len(ground_truth)
    accuracy = metric_accuracy(ground_truth, prediction)

    # confusion matrix
    cf = Counter(zip(ground_truth, prediction))
    categories = set(ground_truth) | set(prediction)
    confusions = {gold: {pred: cf.get((gold, pred), 0) for pred in categories}
                  for gold in categories}
    # marginals
    gold_category_distribution = {
        g: sum([confusions[g][p] for p in categories]) for g in categories
    }
    pred_category_distribution = {
        p: sum([confusions[g][p] for g in categories]) for p in categories
    }

    # kappa
    expected_agreement_cohen = sum([
        (float(gold_category_distribution[c]) / items_count) *
        (float(pred_category_distribution[c]) / items_count)
        for c in categories
    ])
    kappa_cohen = (1.0 * (accuracy - expected_agreement_cohen) /
                   (1 - expected_agreement_cohen))
    return kappa_cohen, expected_agreement_cohen, accuracy


def metric_pk(forest1, forest2):
    masses1 = [get_untyped_masses(tree) for tree in forest1]
    masses2 = [get_untyped_masses(tree) for tree in forest2]
    segments1 = list(chain.from_iterable(masses1))
    segments2 = list(chain.from_iterable(masses2))
    score = segeval.pk(segments1, segments2) * 100
    return score


def metric_windiff(forest1, forest2):
    masses1 = [get_untyped_masses(tree) for tree in forest1]
    masses2 = [get_untyped_masses(tree) for tree in forest2]
    segments1 = list(chain.from_iterable(masses1))
    segments2 = list(chain.from_iterable(masses2))
    score = segeval.window_diff(segments1, segments2) * 100
    return score


def metric_pi_bed(forest1, forest2):
    dataset = {id_: {"1": get_untyped_masses(tree1),
                     "2": get_untyped_masses(tree2)}
               for id_, (tree1, tree2) in enumerate(zip(forest1, forest2))}
    score = segeval.fleiss_pi_linear(dataset) * 100
    return score


def metric_f1(forest1, forest2):
    p = parseval.precision(forest1, forest2)
    r = parseval.recall(forest1, forest2)
    score = f1(p, r) * 100
    return score


def metric_lf1(forest1, forest2):
    lp = parseval.labeled_precision(forest1, forest2)
    lr = parseval.labeled_recall(forest1, forest2)
    score = f1(lp, lr) * 100
    return score


def export_to_dk_agree_unit(annotator, spans, offset=0, typed=True):
    for start, length, cat in spans:
        if not typed:
            cat = None
        yield (offset + start,
               length,
               annotator,
               cat)


def metric_alpha_unit(forest1, forest2, typed=True):
    r = []
    total_length = 0
    # extract spans
    for tree1, tree2 in zip(forest1, forest2):
        spans1, length1 = get_typed_nonoverlapping_spans(tree1)
        spans2, length2 = get_typed_nonoverlapping_spans(tree2)
        r.extend(export_to_dk_agree_unit(0, spans1, offset=total_length,
                                         typed=typed))
        r.extend(export_to_dk_agree_unit(1, spans2, offset=total_length,
                                         typed=typed))
        total_length += length1
    # write to tmp file
    here = os.path.dirname(__file__)
    fout = os.path.join(here, 'alpha', 'agree.tsv')
    with open(fout, 'w') as f:
        f.write('\n'.join('\t'.join(str(e) for e in x) for x in r) + '\n')
    # evaluate with dkpro script
    script = os.path.join(here, 'alpha', 'eval_alpha_unit.sh')
    output = subprocess.check_output(
        script + ' ' + fout, shell=True)
    score = float(output.splitlines()[-1]) * 100
    return score


def metric_alpha_unit_untyped(forest1, forest2):
    return metric_alpha_unit(forest1, forest2, typed=False)


def export_to_gamma(annotator, spans, offset=0):
    for start, length, cat in spans:
        yield (annotator,
               cat,
               offset + start,
               offset + start + length - 1)
