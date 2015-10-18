#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Module providing data and routines for rule-based discourse segmentation

Variables and Constants:
PARENS - dictionary of opening and closing parentheses
DASHES - dictionary of possible dash characters
QUOTES - dictionary of opening and closing quotation mark
DELIMS - dictionary of possible punctuation delimiters
DELIM_NAMES - symbolic names of punctuation delimiters
reporting_verbs - list of verbs that instantiate direct speech
dass_verbs - list of verbs that might introduce a direct dass-object
finite_verbs - Trie of finite German verbs
discourse_preps - Trie of discourse connectives

Methods:
data_dir - partial function to obtain the relative location of data files
load_verb_list - function for reading lists of words from file and putting them
                 into trie

@author = Jean VanCoppenolle, Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <vancoppenolle at uni dash potsdam dot de>, <sidarenk at uni dash potsdam dot de>
"""

##################################################################
# Imports
from .util import Trie, VerbMatcher

from functools import partial
import codecs
import os

##################################################################
# Methods
data_dir = partial(os.path.join, os.path.dirname(__file__), 'data')

def load_verb_list(filename):
    """
    Read list of words from file and generate a trie

    @param filename - name of the file to read from

    @return trie of words present in input file
    """
    verbs = set()
    with codecs.open(data_dir(filename), encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            verbs.add(line)
    return VerbMatcher(verbs)

##################################################################
# Variables and constants
PARENS = {
    u'(': u')',
    u'[': u']',
    u'{': u'}',
    u'<': u'>',
}

DASHES = {
    u'-': u'-',
    u'\u2013': u'\u2013',  # en-dash
    u'\u2014': u'\u2014',  # em-dash
    u'\u2015': u'\u2015',  # horizontal bar
}

PARENS.update(DASHES)

QUOTES = {
    u"'": u"'",  # single quotes
    u'"': u'"',  # double quotes
    u'„': u'"',  # double quotes
    u'‌«': u'»',  # guillemet (angle quotes)
    u'“': u'”',  # quotation marks
    u'‘': u'’',  # inverted commas
}

DELIMS = {}
DELIMS.update(PARENS)
DELIMS.update(DASHES)
DELIMS.update(QUOTES)

DELIM_NAMES = {
    u'(': 'L_PAREN',
    u')': 'R_PAREN',
    u'[': 'L_SQUARE_BRACKET',
    u']': 'R_SQUARE_BRACKET',
    u'{': 'L_CURLY_BRACKET',
    u'}': 'R_CURLY_BRACKET',
    u'<': 'LT',
    u'>': 'GT',
    u'-': 'FIGURE_DASH',
    u'\u2013': 'EN_DASH',
    u'\u2014': 'EM_DASH',
    u'\u2015': 'HORIZONTAL_BAR',
    u"'": 'SINGLE_QUOTE',
    u'"': 'DOUBLE_QUOTE',
    u'‌«': 'L_ANGLE_QUOTE',
    u'»': 'R_ANGLE_QUOTE',
    u'“': 'QUOTATION_MARK',
    u'”': 'QUOTATION_MARK',
    u'„': 'QUOTATION_MARK',
    u'‘': 'L_INVERTED_COMMA',
    u'’': 'R_INVERTED_COMMA',
}

reporting_verbs = load_verb_list('reporting_verbs.txt')
dass_verbs = load_verb_list('dass_verbs.txt')

finite_verbs = Trie()
with codecs.open(data_dir('finite_verbs.txt'), encoding='utf-8') as _fp:
    for _line in _fp:
        _line = _line.strip()
        if not _line:
            continue
        (_verb, _type) = _line.split('\t', 1)
        finite_verbs.add_word(_verb, _type)
    del _line, _verb, _type
del _fp

discourse_preps = Trie()
with codecs.open(data_dir('discourse_preps.txt'), encoding='utf-8') as _fp:
    for _line in _fp:
        _line = _line.strip()
        if not _line:
            continue
        try:
            _prep, _circumpos = [s.strip() for s in _line.split('*')]
        except ValueError:
            _prep, _circumpos = _line, None
        discourse_preps.add_word(_prep, _circumpos)
    del _line, _prep, _circumpos
del _fp
