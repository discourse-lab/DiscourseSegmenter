#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""Module defining methods common to many modules.

Attributes:
  _ispunct (method): check if word consists only of punctuation characters
  prune_punc (method): remove tokens representing punctuation from set
  read_segments (method): default method for reading segment files
  score_substitute (method): custom weighting function used for token alignment
  translate_toks (method): replace tokens and return updated set

"""

##################################################################
# Imports
import string


##################################################################
# Constants
DEPS = "deps"
NONE = str(None)
REL = "rel"
TAG = "tag"
WORD = "word"


##################################################################
# Methods
def _ispunct(a_word):
    """Check if word consists only of punctuation characters.

    Args:
      a_word (str): word to check

    Returns:
      bool: True if word consists only of punctuation characters,
        False otherwise

    """
    return all(c in string.punctuation for c in a_word)


def prune_punc(a_toks):
    """Remove tokens representing punctuation from set.

    Args:
      a_toks (iterable): original tokens

    Returns:
      frozenset: tokens without punctuation marks

    """
    return frozenset([tok for tok in a_toks if not _ispunct(tok[-1])])


def read_segments(a_lines):
    """Read file and return a list of segment dictionaries.

    Args:
      a_lines (list): decoded lines of the input file

    Returns:
      dict: mapping from tokens to segments

    """
    segs2toks = {}
    s_c = t_c = 0
    tokens = []
    atoks = []
    new_seg = None
    active_tokens = set()
    active_segments = []
    # read segments
    for iline in a_lines:
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
                assert active_segments, \
                    "Unbalanced closing parenthesis at line: " + repr(iline)
                active_tokens = set(atoks)
                del atoks[:]
                for a_s in active_segments:
                    segs2toks[a_s].update(active_tokens)
                active_segments.pop()
                continue
            else:
                atoks.append((t_c, tok))
                t_c += 1
        assert not active_segments, \
            "Unbalanced opening parenthesis at line: " + repr(iline)
    toks2segs = dict()
    segments = segs2toks.keys()
    segments.sort(key=lambda el: el[0])
    for seg in segments:
        toks = frozenset(segs2toks[seg])
        # it can be same tokenset corresponds to multiple segments, in that
        # case we leave the first one that we encounter
        if toks in toks2segs:
            continue
        assert toks not in toks2segs, \
            "Multiple segments correspond to the same tokenset: '" + \
            repr(toks) + "': " + repr(seg) + ", " + repr(toks2segs[toks])
        toks2segs[toks] = seg
    return toks2segs


def score_substitute(a_c1, a_c2):
    """Score substitution of two characters.

    Args:
      a_c1 (str): first word to compare
      a_c2 (str): second word to compare

    Returns:
      int: 2 if the last characters of both words are equal, -3 otherwise

    """
    return 2 if a_c1[-1] == a_c2[-1] else -3


def translate_toks(a_toks, a_translation):
    """Translate tokens and return translated set.

    Args:
      a_toks (iterable): tokens to be translated
      a_translation (dict): - translation dictionary for tokens

    Returns:
      frozenset: translated tokens

    """
    if a_translation is None:
        return a_toks
    ret = set()
    for tok in a_toks:
        for t_tok in a_translation[tok]:
            ret.add(t_tok)
    return frozenset(ret)
