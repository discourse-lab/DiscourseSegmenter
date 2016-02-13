#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""
Calculates parsing evaluation metrics: precision, recall, labeled precision and
labeled recall.

Code originally from Mike Gasser's course 'Natural Language
Processing', Indiana University, Fall 2010. Modifications by Alexander
Koller, 'Computerlinguistische Techniken', University of Potsdam,
Summer 2011.
"""

##################################################################
# Imports
from nltk.tree import Tree
import copy


##################################################################
# Methods
def precision(golds, parses, ignore_labels=True):
    """Return the proportion of brackets in the suggested parse tree that are
    in the gold standard. Parameter golds is a list of NLTK Tree
    objects (= the gold standard from the corpus). Parameter parses is
    a list whose elements are either NLTK Trees, or None in case there
    was no parse for a given input. Use ignore_labels to choose
    labeled or unlabeled precision. """

    total = 0
    successes = 0

    for (gold, parse) in zip(golds, parses):
        if parse is not None:
            parsebrackets = list_brackets(parse)
            goldbrackets = list_brackets(gold)

            parsebrackets_u = list_brackets(parse, ignore_labels=True)
            goldbrackets_u = list_brackets(gold, ignore_labels=True)

            if ignore_labels:
                candidate = parsebrackets_u
                gold = goldbrackets_u
            else:
                candidate = parsebrackets
                gold = goldbrackets

            total += len(candidate)
            for bracket in candidate:
                if bracket in gold:
                    successes += 1

    return float(successes) / float(total)


def recall(golds, parses, ignore_labels=True):
    """Return the proportion of brackets in the gold standard that are in the
    suggested parse tree. See precision for details."""

    total = 0
    successes = 0

    for (gold, parse) in zip(golds, parses):
        goldbrackets = list_brackets(gold)
        goldbrackets_u = list_brackets(gold, ignore_labels=True)
        if ignore_labels:
            gold = goldbrackets_u
        else:
            gold = goldbrackets

        total += len(gold)

        if parse is not None:
            parsebrackets = list_brackets(parse)
            parsebrackets_u = list_brackets(parse, ignore_labels=True)

            if ignore_labels:
                candidate = parsebrackets_u
            else:
                candidate = parsebrackets

            for bracket in gold:
                if bracket in candidate:
                    successes += 1

    return float(successes) / float(total)


def labeled_precision(gold, parse):
    return precision(gold, parse, ignore_labels=False)


def labeled_recall(gold, parse):
    return recall(gold, parse, ignore_labels=False)


def words_to_indexes(tree):
    """Return a new tree based on the original tree, such that the leaf values
    are replaced by their indexs."""

    out = copy.deepcopy(tree)
    leaves = out.leaves()
    for index in range(0, len(leaves)):
        path = out.leaf_treeposition(index)
        out[path] = index + 1
    return out


def firstleaf(tr):
    return tr.leaves()[0]


def lastleaf(tr):
    return tr.leaves()[-1]


def list_brackets(tree, ignore_labels=False):
    tree = words_to_indexes(tree)

    def not_pos_tag(tr):
        return tr.height() > 2

    def label(tr):
        if ignore_labels:
            return "ignore"
        else:
            return tr.label()

    subtrees = tree.subtrees(filter=not_pos_tag)
    return [(firstleaf(sub), lastleaf(sub), label(sub)) for sub in subtrees]


def myparse(s):
    return Tree.fromstring(s)


def example1():
    gold = myparse(
        """
(PX
    (PX
        (APPR an)
        (NX
            (ART einem)
            (NX
                (NX (NN Samstag))
                (KON oder)
                (NX (NN Sonntag)))))
    (ADVX (ADV vielleicht)))
    """)

    parse = myparse(
        """
(PX
    (PX
        (APPR an)
        (NX
            (ART einem)
            (NN Samstag)))
    (NX (KON oder) (NX (NN Sonntag)))
    (ADVX (ADV vielleicht)))
""")

    pscore = precision([gold], [parse])
    rscore = recall([gold], [parse])
    print pscore, rscore


def example2():
    gold = myparse(
        """
(SIMPX
    (VF
        (PX (appr von)
            (NX
                (pidat allen)
                (ADJX (adja kulturellen))
                (nn Leuchttuermen))))
    (LK
        (VXFIN (vxfin besitzt)))
    (MF
        (ADVX
            (ADVX (adv nach))
            (kon wie)
            (ADVX (adv vor)))
        (NX
            (art das)
            (nn Theater))
        (NX
            (art das)
            (ADJX (adja unsicherste))
            (nn Fundament))))
""")

    parse = myparse(
        """
(R-SIMPX
    (LV
        (PX
            (appr von)
            (NX (pidat allen))))
    (VF
        (NX
            (ADJX (adja kulturellen))
            (nn Leuchttuermen)))
    (LK (VXFIN (vvfin besitzt)))
    (MF
        (PX
            (PX (appr nach))
            (kon wie)
            (PX
                (appr vor)
                (NX
                    (art das)
                    (nn Theater))))
        (NX
            (art das)
            (ADJX (adja unsicherste))
            (nn Fundament))))
""")
    print "Precision:", precision([gold], [parse])
    print "Labeled precision:", labeled_precision([gold], [parse])
    print "Recall:", recall([gold], [parse])
    print "Labeled recall:", labeled_recall([gold], [parse])

    print "SWITCH GOLD PARSE:"
    print "Precision:", precision([parse], [gold])
    print "Labeled precision:", labeled_precision([parse], [gold])
    print "Recall:", recall([parse], [gold])
    print "Labeled recall:", labeled_recall([parse], [gold])


def main():
    example1()
    example2()

##################################################################
# Main
if __name__ == "__main__":
    main()
