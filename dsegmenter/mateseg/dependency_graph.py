#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
Created on 03.01.2015

@author: Andreas Peldszus
'''

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from dsegmenter.common import DEPS, REL, TAG, WORD

from nltk.parse.dependencygraph import DependencyGraph as NLTKDependencyGraph
import sys

##################################################################
# Constants
HEAD = "head"
ADDRESS = "address"
LEMMA = "lemma"
CTAG = "ctag"
FEATS = "feats"

STANDARD_FIELDS = [HEAD, DEPS, WORD, REL, TAG, ADDRESS, LEMMA, CTAG, FEATS]

TOP_TAG_LABEL = 'TOP'
TOP_RELATION_LABEL = 'ROOT'
'''The label of relations to the root node in the mate dep. parser output.'''


##################################################################
# Class
class DependencyGraph(NLTKDependencyGraph):

    def words(self):
        '''yields all words except the implicit root node in linear order'''
        for address, node in sorted(self.nodes.items()):
            if node[TAG] != TOP_TAG_LABEL:
                yield node[WORD]

    def subgraphs(self, exclude_root=False):
        '''yields all nodes in linear order'''
        for address, node in sorted(self.nodes.items()):
            if exclude_root and node[TAG] == TOP_TAG_LABEL:
                continue
            else:
                yield node

    def get_dependencies_simple(self, address):
        '''returns a sorted list of the addresses of all dependencies of the
           node at the specified address'''
        deps_dict = self.nodes[address].get(DEPS, {})
        return sorted([e for l in deps_dict.values() for e in l])

    def address_span(self, start_address):
        '''returns the addresses of nodes (im)mediately depending on the given
           starting address in a dependency graph, except for the root node'''
        worklist = [start_address]
        addresses = []
        while len(worklist) != 0:
            address = worklist.pop(0)
            addresses.append(address)
            for _rel, deps in self.nodes[address][DEPS].items():
                worklist.extend(deps)
        return sorted(addresses)

    def token_span(self, start_address=0):
        '''returns the words (im)mediately depending on the given address in a
           dependency graph in correct linear order, except for the root node
        '''
        addresses = self.address_span(start_address)
        return [self.nodes[address][WORD]
                for address in sorted(addresses) if address != 0]

    def is_valid_parse_tree(self):
        '''check structural integrity of the parse;
           for the moment just check for a unique root'''
        root = self.get_dependencies_simple(0)
        if len(root) < 1:
            print("Warning: No root address", file=sys.stderr)
            return False
        if len(root) > 1:
            print("Warning: More than one root address", file=sys.stderr)
            return False
        return True

    def length(self):
        '''returns the length in tokens, i.e. the number of nodes excluding
           the artifical root'''
        return len(self.nodes) - 1

    def annotate(self, iterable, field_name):
        '''annotate the nodes (excluding the artifical root) with an additional
           non-standard field, the values being provided in an iterable in
           linear order corresponding to the node order'''
        assert len(iterable) == self.length()
        assert field_name not in STANDARD_FIELDS
        for i, value in enumerate(iterable, 1):
            self.nodes[i][field_name] = value

    def deannotate(self, field_name):
        '''remove annotations of an additional non-standard field'''
        assert field_name not in STANDARD_FIELDS
        for node in self.nodes.values():
            if field_name in node:
                del node[field_name]


##################################################################
# Methods
def transform_line(line):
    """Transform a mate line to a valid conll 2007 format.

    Args:
      a_line (str): input line to transform

    Returns:
      str: transformed line

    """
    f = line.split('\t')
    # escape parenthesis
    token = f[1]
    if token == '(':
        token = '-OP-'
    elif token == ')':
        token = '-CP-'
    # The nltk v3 implementation of dependency graphs needs an explicit
    # root relation label. Mate's output uses '--' as a label for relations
    # to the root, but also for punctuations. We thus translate the
    # relation label to 'ROOT'.
    if f[9] == '0':
        f[11] = TOP_RELATION_LABEL
    return '\t'.join([f[0], token, f[3], f[5], f[5], f[7], f[9], f[11],
                      '_', '_'])


def number_tokens(dgraph):
    """Prefix all tokens in dependency graphs with their running number.

    Args:
      dgraph (nltk.parse.dependencygraph.DependencyGraph):
        list of dependency trees

    Returns:
      nltk.parse.dependencygraph.DependencyGraph:
        dependency trees with numbered tokens

    """
    cnt = 0
    for node in dgraph.subgraphs(exclude_root=True):
        node[WORD] = (cnt, node[WORD])
        cnt += 1
    return dgraph


def tree2tok(a_tree, a_tree_idx, a_root_idx, a_tk_start=0):
    """Create dictionary mapping dependency trees to numbered tokens.

    Args:
      a_tree (DependencyGraph): tree to analyze
      a_tree_idx (int): tree index in the document
      a_root_idx (int): index of the root node
      a_tk_start (int): starting position of the first token

    Returns:
      (dict) mapping from subtrees to their yields

    """
    # set of terminals corresponding to the given node
    iroot = a_tree.nodes[a_root_idx]
    tkset = set()
    if iroot[WORD] is not None:
        tkset.add((a_tk_start + iroot[WORD][0], iroot[WORD][1]))
    tr2tk = {(a_tree_idx, a_root_idx): (a_tree, tkset)}
    for ch_idcs in iroot[DEPS].itervalues():
        for ch_idx in ch_idcs:
            t2t = tree2tok(a_tree, a_tree_idx, ch_idx, a_tk_start)
            tr2tk.update(t2t)
            tkset.update(t2t[(a_tree_idx, ch_idx)][-1])
    return tr2tk


def read_tok_trees(a_lines):
    """Read file and return a mapping from tokens to trees and a list of trees.

    Args:
      a_lines (list[str]): decoded lines of the input file

    Returns:
      2-tuple: list of dictionaries mapping tokens to trees and a list of trees

    """
    toks = []
    tok_c = 0
    t2t = None
    trees = [t for t in read_trees(a_lines)]
    trees2toks = dict()

    for i, itree in enumerate(trees):
        t2t = tree2tok(itree, i, 0, tok_c)
        trees2toks.update(t2t)
        # increment token counter by the number of tokens in the sentence
        tok_c += len(t2t[(i, 0)][-1])

    toks2trees = dict()
    for ((tree_c, tree_pos), (tree, toks)) in trees2toks.iteritems():
        # skip the abstract root node
        if tree_pos == 0:
            continue
        toks = frozenset(toks)
        if toks in toks2trees:
            toks2trees[toks].append((tree, tree_pos))
        else:
            toks2trees[toks] = [(tree, tree_pos)]
    return (toks2trees, trees)


def read_trees(a_lines):
    """Read file and yield DependencyGraphs.

    Args:
      a_lines (list[str]): iterable over decoded lines of the input file

    Yields:
      nltk.parse.dependencygraph.DependencyGraph:

    """
    toks = []
    for iline in a_lines:
        iline = iline.strip()
        if not iline:
            if toks:
                yield number_tokens(
                    DependencyGraph('\n'.join(toks),
                                    top_relation_label=TOP_RELATION_LABEL))
                del toks[:]
        else:
            toks.append(transform_line(iline))
    if toks:
        yield number_tokens(
                    DependencyGraph('\n'.join(toks),
                                    top_relation_label=TOP_RELATION_LABEL))
