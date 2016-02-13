#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""

Various methods for extracting spans and segmentations from trees.

@author = Andreas Peldszus
@mail = <peldszus at uni dash potsdam dot de>
@version = 0.1.0

"""

##################################################################
# Imports
from nltk.tree import Tree


##################################################################
# Methods
def get_typed_spans(tree):
    """ Extracts the token spans from a tree.

    Returns a list of span tuples (position, length, label) for
    each node in the tree, as well as the full length of the input tree.

    >>> from nltk.tree import Tree
    >>> get_typed_spans(Tree.fromstring('(A x x x)'))
    ([(0, 3, 'A')], 3)
    >>> get_typed_spans(Tree.fromstring('(A x (B x (C x)))'))
    ([(0, 3, 'A'), (1, 2, 'B'), (2, 1, 'C')], 3)
    >>> get_typed_spans(Tree.fromstring('(A (B (C x) x) x)'))
    ([(0, 1, 'C'), (0, 2, 'B'), (0, 3, 'A')], 3)
    >>> get_typed_spans(Tree.fromstring('(A (B x) (C x) (D x))'))
    ([(0, 1, 'B'), (0, 3, 'A'), (1, 1, 'C'), (2, 1, 'D')], 3)
    >>> get_typed_spans(Tree.fromstring('(A x (B x (C x) x) x)'))
    ([(0, 5, 'A'), (1, 3, 'B'), (2, 1, 'C')], 5)
    """
    spans, length = _get_typed_spans(tree)
    return sorted(spans), length


def _get_typed_spans(tree, start=0):
    spans = []
    length = 0
    for child in tree:
        if isinstance(child, Tree):
            child_spans, child_length = \
                _get_typed_spans(child, start=start+length)
            spans.extend(child_spans)
            length += child_length
        else:
            length += 1
    spans.append((start, length, tree.label()))
    return spans, length


def get_untyped_spans(tree):
    """ Extracts the untyped token spans from a tree.

    Returns a list of span tuples (position, length) for
    each node in the tree, as well as the full length of the input tree.

    >>> from nltk.tree import Tree
    >>> get_untyped_spans(Tree.fromstring('(A x x x)'))
    ([(0, 3)], 3)
    >>> get_untyped_spans(Tree.fromstring('(A x (B x (C x)))'))
    ([(0, 3), (1, 2), (2, 1)], 3)
    >>> get_untyped_spans(Tree.fromstring('(A (B (C x) x) x)'))
    ([(0, 1), (0, 2), (0, 3)], 3)
    >>> get_untyped_spans(Tree.fromstring('(A (B x) (C x) (D x))'))
    ([(0, 1), (0, 3), (1, 1), (2, 1)], 3)
    >>> get_untyped_spans(Tree.fromstring('(A x (B x (C x) x) x)'))
    ([(0, 5), (1, 3), (2, 1)], 5)
    """
    spans, length = _get_typed_spans(tree)
    return sorted([(p, l) for p, l, _ in spans]), length


def get_typed_masses(tree):
    """ Extracts the typed segmentation masses from a tree.

    Extracts a segmentation of the leaves of a tree, such that each
    start or end of a child node triggers a boundary. The segmentation
    is returned as a list of segments tuples (length, type).

    >>> from nltk.tree import Tree
    >>> get_typed_masses(Tree.fromstring('(A x x x)'))
    [(3, 'A')]
    >>> get_typed_masses(Tree.fromstring('(A x (B x (C x)))'))
    [(1, 'A'), (1, 'B'), (1, 'C')]
    >>> get_typed_masses(Tree.fromstring('(A (B (C x) x) x)'))
    [(1, 'C'), (1, 'B'), (1, 'A')]
    >>> get_typed_masses(Tree.fromstring('(A (B x) (C x) (D x))'))
    [(1, 'B'), (1, 'C'), (1, 'D')]
    >>> get_typed_masses(Tree.fromstring('(A x (B x (C x) x) x)'))
    [(1, 'A'), (1, 'B'), (1, 'C'), (1, 'B'), (1, 'A')]
    """
    masses = []
    length = 0
    for child in tree:
        if isinstance(child, Tree):
            if length > 0:
                masses.append((length, tree.label()))
            child_masses = get_typed_masses(child)
            masses.extend(child_masses)
            length = 0
        else:
            length += 1
    if length > 0:
        masses.append((length, tree.label()))
    return masses


def get_untyped_masses(tree):
    """ Extracts the untyped segmentation masses from a tree.

    Extracts a segmentation of the leaves of a tree, such that each
    start or end of a child node triggers a boundary. The segmentation
    is returned as a list of segments lengths.

    >>> from nltk.tree import Tree
    >>> get_untyped_masses(Tree.fromstring('(A x x x)'))
    [3]
    >>> get_untyped_masses(Tree.fromstring('(A x (B x (C x)))'))
    [1, 1, 1]
    >>> get_untyped_masses(Tree.fromstring('(A (B (C x) x) x)'))
    [1, 1, 1]
    >>> get_untyped_masses(Tree.fromstring('(A (B x) (C x) (D x))'))
    [1, 1, 1]
    >>> get_untyped_masses(Tree.fromstring('(A x (B x (C x) x) x)'))
    [1, 1, 1, 1, 1]
    """
    return list(m for m, _c in get_typed_masses(tree))


def typed_masses_to_spans(typed_masses):
    """ Converts typed segmentations in the masses format to spans.

    The resulting spans are non-overlapping and in linear order.

    >>> list(typed_masses_to_spans([(1, 'A'), (1, 'B'), (1, 'C')]))
    [(0, 1, 'A'), (1, 1, 'B'), (2, 1, 'C')]
    >>> list(typed_masses_to_spans([(3, 'A'), (2, 'B'), (1, 'C')]))
    [(0, 3, 'A'), (3, 2, 'B'), (5, 1, 'C')]
    >>> list(typed_masses_to_spans([(1, 'A'), (2, 'B'), (3, 'C')]))
    [(0, 1, 'A'), (1, 2, 'B'), (3, 3, 'C')]
    """
    start = 0
    for length, cat in typed_masses:
        yield (start, length, cat)
        start += length


def get_typed_nonoverlapping_spans(tree):
    """ Extracts non-overlapping typed spans from a tree.

    Returns a linearly ordered list of pre-terminal nodes as span
    tuples (position, length, label).

    >>> from nltk.tree import Tree
    >>> get_typed_nonoverlapping_spans(Tree.fromstring('(A x x x)'))
    ([(0, 3, 'A')], 3)
    >>> get_typed_nonoverlapping_spans(Tree.fromstring('(A x (B x (C x)))'))
    ([(0, 1, 'A'), (1, 1, 'B'), (2, 1, 'C')], 3)
    >>> get_typed_nonoverlapping_spans(Tree.fromstring('(A (B (C x) x) x)'))
    ([(0, 1, 'C'), (1, 1, 'B'), (2, 1, 'A')], 3)
    >>> get_typed_nonoverlapping_spans(
    ...     Tree.fromstring('(A (B x) (C x) (D x))'))
    ([(0, 1, 'B'), (1, 1, 'C'), (2, 1, 'D')], 3)
    >>> get_typed_nonoverlapping_spans(
    ...     Tree.fromstring('(A x (B x (C x) x) x)'))
    ([(0, 1, 'A'), (1, 1, 'B'), (2, 1, 'C'), (3, 1, 'B'), (4, 1, 'A')], 5)
    """
    typed_masses = get_typed_masses(tree)
    spans = list(typed_masses_to_spans(typed_masses))
    last_start, last_length, _ = spans[-1]
    return spans, last_start + last_length


def get_untyped_nonoverlapping_spans(tree):
    """ Extracts non-overlapping un-typed spans from a tree.

    Returns a linearly ordered list of pre-terminal nodes as span
    tuples (position, length).

    >>> from nltk.tree import Tree
    >>> get_untyped_nonoverlapping_spans(Tree.fromstring('(A x x x)'))
    ([(0, 3)], 3)
    >>> get_untyped_nonoverlapping_spans(Tree.fromstring('(A x (B x (C x)))'))
    ([(0, 1), (1, 1), (2, 1)], 3)
    >>> get_untyped_nonoverlapping_spans(Tree.fromstring('(A (B (C x) x) x)'))
    ([(0, 1), (1, 1), (2, 1)], 3)
    >>> get_untyped_nonoverlapping_spans(
    ...     Tree.fromstring('(A (B x) (C x) (D x))'))
    ([(0, 1), (1, 1), (2, 1)], 3)
    >>> get_untyped_nonoverlapping_spans(
    ...     Tree.fromstring('(A x (B x (C x) x) x)'))
    ([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)], 5)
    """
    spans, length = get_typed_nonoverlapping_spans(tree)
    return list((s, l) for s, l, _c in spans), length


def get_confusions(tree1, tree2):
    """ Extract pairs of types of matching spans in the input trees

    Compares the spans found in two trees. A span from one tree is
    considered to be matching one in the other tree, if they have the
    same starting position and length. In this case, a pair with the
    type of this span in both trees is yielded. Spans from one tree
    without a corresponding span in the other tree yield a pair of types,
    where the type for the tree missing the span is None. Overlaps,
    entailments and near misses are not considered. The leaves are ignored.

    >>> sorted(get_confusions(Tree.fromstring('(A x (B x (C x)))'),
    ...                       Tree.fromstring('(A x (B x (C x)))')))
    [('A', 'A'), ('B', 'B'), ('C', 'C')]
    >>> sorted(get_confusions(Tree.fromstring('(A x (B x (C x)))'),
    ...                       Tree.fromstring('(A x (D x (E x)))')))
    [('A', 'A'), ('B', 'D'), ('C', 'E')]
    >>> sorted(get_confusions(Tree.fromstring('(A x x (B x))'),
    ...                       Tree.fromstring('(A x (B x x))')))
    [(None, 'B'), ('A', 'A'), ('B', None)]
    """
    spans1 = {(s, l): c for s, l, c in get_typed_spans(tree1)[0]}
    spans2 = {(s, l): c for s, l, c in get_typed_spans(tree2)[0]}

    for (start, length), cat1 in spans1.iteritems():
        if (start, length) in spans2:
            cat2 = spans2[(start, length)]
            yield cat1, cat2
        else:
            yield cat1, None

    for (start, length), cat2 in spans2.iteritems():
        if (start, length) not in spans1:
            yield None, cat2
