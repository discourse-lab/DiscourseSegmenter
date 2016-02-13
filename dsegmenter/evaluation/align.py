#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""

Aligning two differently tokenized string representations of bracket
trees.

@author = Andreas Peldszus
@mail = <peldszus at uni dash potsdam dot de>
@version = 0.1.0

"""


##################################################################
# Exceptions
class AlignmentError(Exception):
    pass


##################################################################
# Methods
def align_tokenized_tree(toks1, toks2, tree_pair_name="no-name-given",
                         delexicalize=False):
    """Aligns two bracket trees with different tokenizations.

    This function aligns two tokenized bracket tree strings. Brackets are
    assumed to be round. Opening brackets and nodelabels are assumed to be
    one token of the form '(X', closing brackets are assumed to be separate
    tokens. The alignment function returns the tokenized strings, modified
    such that the tokens align.

    >>> align_tokenized_tree('(S a b c )'.split(), '(S a b c )'.split())
    (0, ['(S', 'a', 'b', 'c', ')'], ['(S', 'a', 'b', 'c', ')'])

    >>> align_tokenized_tree('(S (S (S a b ) ) )'.split(), '(S a b )'.split())
    (0, ['(S', '(S', '(S', 'a', 'b', ')', ')', ')'], ['(S', 'a', 'b', ')'])

    >>> align_tokenized_tree('(S a (X b (Y c ) ) )'.split(),
    ...                      '(S a b c )'.split())
    (0, ['(S', 'a', '(X', 'b', '(Y', 'c', ')', ')', ')'], ['(S', 'a', 'b', 'c',
    ')'])

    >>> align_tokenized_tree('(S a b c )'.split(), '(S ab c )'.split())
    Aligning subtokens a and ab in text no-name-given ...
    (0, ['(S', 'a', 'b', 'c', ')'], ['(S', 'a', 'b', 'c', ')'])

    >>> align_tokenized_tree('(S ab c )'.split(), '(S a bc )'.split())
    Aligning subtokens ab and a in text no-name-given ...
    Aligning subtokens b and bc in text no-name-given ...
    (0, ['(S', 'a', 'b', 'c', ')'], ['(S', 'a', 'b', 'c', ')'])

    >>> align_tokenized_tree('(S a b c )'.split(), '(S a )'.split())
    Traceback (most recent call last):
        ...
    AlignmentError: Error: Overlap. remaining_a=['c', ')'], remaining_b=[]

    >>> align_tokenized_tree('(S a b c )'.split(), '(S a c )'.split())
    Traceback (most recent call last):
        ...
    AlignmentError: Error: Trees don't align. current_a, current_b, a, b:
    ['(S', 'a'] b ['c', ')']
    ['(S', 'a'] c [')']

    """
    # make sure we don't get empty strings
    a = [i.strip() for i in toks1 if i.strip() != '']
    b = [i.strip() for i in toks2 if i.strip() != '']

    # initialize variables used while aligning the trees
    new_a = []
    new_b = []
    current_a = None
    current_b = None
    buffer_a = ''
    buffer_b = ''
    error = 0

    def add_to(new_list, token):
        if delexicalize:
            token = 'tok'
        new_list.append(token)

    while True:
        # if nothing in buffer a, get next symbol in a, skipping treebrackets
        while buffer_a == '':
            if len(a) > 0:
                current_a = a.pop(0)
                if current_a.startswith('(') or current_a == ')':
                    new_a.append(current_a)
                else:
                    break
            else:
                current_a = None
                break
        else:
            current_a = buffer_a
            buffer_a = ''

        # if nothing in buffer b, get next symbol in b, skipping treebrackets
        while buffer_b == '':
            if len(b) > 0:
                current_b = b.pop(0)
                if current_b.startswith('(') or current_b == ')':
                    new_b.append(current_b)
                else:
                    break
            else:
                current_b = None
                break
        else:
            current_b = buffer_b
            buffer_b = ''

        # compare symbols
        if current_a is None and current_b is None:
            if [] == a == b:
                # successfully finished
                break
            else:
                # overlap match
                raise AlignmentError(
                    "Error: Overlap. remaining_a=%s, remaining_b=%s" % (a, b))
        elif current_a is None or current_b is None:
            # one string is consumed, the other not,
            # another form of overlap match
            raise AlignmentError(
                "Error: Overlap. remaining_a=%s, remaining_b=%s" % (a, b))
        elif current_a == current_b:
            # align tokens
            add_to(new_a, current_a)
            add_to(new_b, current_b)
        elif current_a.startswith(current_b):
            # align subtokens
            add_to(new_a, current_b)
            add_to(new_b, current_b)
            buffer_a = current_a[len(current_b):]
            print "Aligning subtokens %s and %s in text %s ..." % \
                (current_a, current_b, tree_pair_name)
        elif current_b.startswith(current_a):
            # align subtokens
            add_to(new_a, current_a)
            add_to(new_b, current_a)
            buffer_b = current_b[len(current_a):]
            print "Aligning subtokens %s and %s in text %s ..." % \
                (current_a, current_b, tree_pair_name)
        else:
            # cannot align
            raise AlignmentError(
                "Error: Trees don't align. current_a, current_b, a, b:\n" +
                "%s %s %s\n" % (new_a[-5:], current_a, a[:5]) +
                "%s %s %s" % (new_b[-5:], current_b, b[:5]))
    return error, new_a, new_b
