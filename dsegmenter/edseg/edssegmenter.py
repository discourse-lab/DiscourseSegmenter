#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing rule-based discourse segmenter `EDSSegmenter`.

Attributes:
  WESWEGEN_SET (set): set of strings representing causal connectives
  SDS_LABEL (str): label for sentence discourse segments
  EDS_LABEL (str): label for elementary discourse segments
  MAIN_CLAUSE (str): label for discourse segments that encompass main clauses
  SUB_CLAUSE (str): label for discourse segments that encompass subordinate
    clauses
  REL_CLAUSE (str): label for discourse segments that encompass restrictive
    relative clauses
  PAREN (str): label for parenthetical discourse segments
  DISCOURSE_PP (str): label for discourse segments formed by prepositional
    phrases
  EDSSegmenter (class): rule-based discourse segmenter

.. moduleauthor:: Jean VanCoppenolle, Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from .clause_segmentation import ClauseSegmenter
from .finitestateparsing import Tree
from .util import StartOfClauseMatcher, Trie
from . import data

import sys

##################################################################
# Constants
WESWEGEN_SET = set(['weswegen', 'weshalb'])
SDS_LABEL = 'SDS'
EDS_LABEL = 'EDS'
MAIN_CLAUSE = 'MainCl'
SUB_CLAUSE = 'SubCl'
REL_CLAUSE = 'RelCl'
PAREN = 'Paren'
DISCOURSE_PP = 'DiPP'


##################################################################
# Classes
class EDSSegmenter(object):
    """Class for perfoming discourse segmentation on CONLL dependency trees.

    Attributes:
      _clause_segmenter: internal worker for doing discourse segmentation
      _clause_discarder: internal automaton which decides if sentence
        shouldn't be processed
      _sent: internal reference to the sentence being processed
      _tokens: internal reference to the list of processed tokens
      segment: perform discourse segmentation of the CONLL sentence

    """

    def __init__(self, a_clause_segmenter=None):
        """
        Class constructor.

        @param a_clause_segmenter - underlying clause segmenter
        """
        if a_clause_segmenter is None:
            self._clause_segmenter = ClauseSegmenter()
        else:
            self._clause_segmenter = a_clause_segmenter
        self._clause_discarder = StartOfClauseMatcher.from_file(
            data.data_dir('skip_rules.txt'))
        self._sent = None
        self._tokens = []

    def segment(self, sent):
        """Segment CONLL trees.

        Args:
          sent (CONLLSentence)L CONLL tree to process

        Returns:
          Segment: sentence-level discourse segment

        """
        self._sent = sent
        clauses = self._clause_segmenter.segment(sent)
        sds = Tree(SDS_LABEL)
        eds = self._make_eds(sds, type=MAIN_CLAUSE)
        for idx, clause in enumerate(clauses):
            eds = self._process_clause(clause, idx, clauses, sds, eds)
        return sds

    def _process_clause(self, clause, idx, clauses, sds, eds, depth=0):
        if not clause:
            return eds
        clause = self._flatten_coord(clause)
        if self._is_embedded(idx, clauses):
            depth += 1
        child1 = clause.first_child
        if not self._is_token(child1) and child1.label == clause.label:
            for idx, child in enumerate(clause):
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
            return eds
        self._tokens, prev_toks = list(clause.iter_terminals()), self._tokens
        if self._clause_discarder.match(self._tokens, prev_toks):
            for idx, child in enumerate(clause):
                if self._is_token(child):
                    eds.append(child)
                else:
                    eds = self._process_clause(child, idx, clause, sds, eds,
                                               depth=depth)
            return eds
        try:
            meth = getattr(self, '_process_{0}'.format(clause.label.lower()))
        except AttributeError:
            return eds
        return meth(clause, idx, clauses, sds, eds, depth=depth)

    def _pairwise(self, a_elements):
        iterable = iter(a_elements)
        prev = next(iterable, None)
        if prev is None:
            return
        yield None, prev
        for elem in iterable:
            yield prev, elem
            prev = elem

    def _process_maincl(self, clause, idx, parent, sds, eds, depth=0):
        verb, deps = self._find_verb_and_dependants(clause)
        if clause.get('makeVerbLess'):
            eds = self._make_eds(sds, type=MAIN_CLAUSE)
        elif verb is None:
            self._flatten(clause)
            eds.extend(clause)
            return eds
        elif (len(eds) and
              not data.reporting_verbs.match(verb['lemma'], deps) and
              not self._is_unintroduced_complement(clause, idx, parent)):
            if depth > 0:
                eds = self._make_embedded_eds(eds, type=MAIN_CLAUSE)
            elif len(eds):
                eds = self._make_eds(sds, type=MAIN_CLAUSE)
        first_token = True
        for idx, child in enumerate(clause):
            if self._is_token(child):
                if first_token:
                    if eds.get('type') != MAIN_CLAUSE:
                        eds = self._make_eds(sds, type=MAIN_CLAUSE)
                    first_token = False
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_sntsubcl(self, clause, idx, parent, sds, eds, depth=0):
        is_complement = False
        for prev, token in self._pairwise(clause.terminals(3)):
            if not self._is_token(token):
                break
            elif token['lemma'] in ('dass', u'daß', 'ob'):
                # comment the if block below, if you want dass-sentences to be
                # considered as separate EDUs
                if prev is None or (prev['pos'] != 'ADV' and prev['lemma'] != 'so'):
                    is_complement = True
                break
            elif token['pos'] in ('KON'):
                is_complement = True
                break
        if not is_complement:
            if depth > 0:
                eds = self._make_embedded_eds(eds)
            elif len(eds):
                eds = self._make_eds(sds)
            eds.set('type', SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_infsubcl(self, clause, idx, parent, sds, eds, depth=0):
        if self._is_nonfin_subord(clause, eds):
            if len(eds):
                eds = self._make_eds(sds)
            eds.set('type', SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return eds

    def _process_relcl(self, clause, idx, parent, sds, eds, depth=0):
        if depth > 0:
            eds = self._make_embedded_eds(eds)
        elif len(eds):
            eds = self._make_eds(sds)
        eds.set('type', REL_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return sds.last_child

    def _process_infcl(self, clause, idx, parent, sds, eds, depth=0):
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        return eds

    def _process_intcl(self, clause, idx, parent, sds, eds, depth=0):
        child = None
        if clause.terminal(0)['form'] in WESWEGEN_SET or \
                (clause.terminal(1)['form'] in WESWEGEN_SET):
            if depth > 0:
                eds = self._make_embedded_eds(eds)
            elif len(eds):
                eds = self._make_eds(sds)
            eds.set('type', SUB_CLAUSE)
        for idx, child in enumerate(clause):
            if self._is_token(child):
                eds.append(child)
            else:
                eds = self._process_clause(child, idx, clause, sds, eds,
                                           depth=depth)
        if self._is_token(child) and child['form'] == ':':
            eds = self._make_eds(sds, type = MAIN_CLAUSE)
        return eds

    def _process_paren(self, clause, idx, parent, sds, eds, depth=0):
        if clause.first_terminal['form'] not in data.QUOTES and \
                any(tok['pos'].startswith('VV')
                    for tok in clause.iter_terminals()):
            if depth > 0:
                eds = self._make_embedded_eds(eds, type=PAREN)
            elif len(eds):
                eds = self._make_eds(sds, type=PAREN)
        self._flatten(clause)
        eds.extend(clause)
        return sds.last_child

    def _process_pc(self, clause, idx, parent, sds, eds, depth=0):
        tokens = list(clause.iter_terminals())
        eds.extend(tokens)
        return eds

    def _process_any(self, clause, idx, parent, sds, eds, depth=0):
        if eds is None:
            eds = self._make_eds(sds, type = MAIN_CLAUSE)
        # if preceding EDS was processed by PC, it might not have the type
        # label
        if "type" not in eds.feats:
            eds.set('type', MAIN_CLAUSE)
        self._flatten(clause)
        eds.extend(clause)
        return eds

    def _is_unintroduced_complement(self, clause, idx, parent):
        prev_clause = parent[idx - 1] if idx else None
        if prev_clause is None:
            return False
        prev_verb, deps = self._find_verb_and_dependants(prev_clause)
        if prev_verb is None:
            return False
        tokens = clause.iter_terminals()
        token1 = next(tokens, {})
        if token1.get('pos') == 'KON':
            return False
        # Test for cases like "..., nämlich dass ..."
        elif (token1.get('pos') != 'ADV' and
              next(tokens, {}).get('lemma') in ('dass', u'daß')):
            return False
        return data.dass_verbs.match(prev_verb['lemma'], deps)

    def _is_nonfin_subord(self, clause, eds):
        # Test for cases like 'zu <adj> ..., um ... (to <adj> to ...)'
        for tok in clause.children(2):
            if 'lemma' in tok and tok['lemma'].lower() == "um":
                return True
        tokens = list(eds.iter_terminals())[-1:-10:-1]
        try:
            for idx, token in enumerate(tokens):
                if token['pos'] in ('ADJD', 'VVPP'):
                    if (tokens[idx - 1]['lemma'] == 'genug' or
                        tokens[idx + 1]['lemma'] in ('genug', 'zu')):
                        return False
        except IndexError:
            pass
        return True

    def _flatten(self, node, parent=None):
        if self._is_token(node):
            return
        for child in node:
            self._flatten(child, parent=node)
        if parent is not None:
            parent.replace(node, *node)

    def _flatten_coord(self, clause):
        if self._is_token(clause):
            return clause
        try:
            child1, child2 = clause.children(2)
        except ValueError:
            return clause
        if self._is_token(child1) or child1.label != clause.label:
            return clause
        elif not self._is_token(child2) or child2['pos'] != 'KON':
            return clause
        new_clause = Tree(clause.label)
        pending_conj = None
        for child in clause:
            if self._is_token(child):
                assert child['pos'] == 'KON'
                pending_conj = child
            elif pending_conj is not None:
                conjunct = child
                child1 = conjunct.first_child
                while not self._is_token(child1):
                    conjunct = child1
                    child1 = child1.first_child
                conjunct.insert(0, pending_conj)
                pending_conj = None
                new_clause.append(child)
            else:
                new_clause.append(child)
        return new_clause

    def _is_token(self, node):
        return node is not None and not isinstance(node, Tree)

    def _is_embedded(self, idx, parent):
        if self._is_token(parent[idx]) or not idx:
            return False
        return (self._is_token(parent[idx - 1]) and
                self._is_token(parent[idx + 1]))

    def _find_verb_and_dependants(self, clause):
        verb = clause.get('verb')
        if verb is None:
            return None, []
        deps = []
        for token in clause.iter_terminals():
            if token['phead'] == verb['idx']:
                deps.append(token)
        return verb, deps

    def _make_eds(self, sds, **feats):
        sds.append(Tree(EDS_LABEL, feats = feats))
        return sds.last_child

    def _make_embedded_eds(self, eds, **feats):
        feats['embedded'] = True
        embed_eds = Tree(EDS_LABEL, feats=feats)
        eds.append(embed_eds)
        return embed_eds
