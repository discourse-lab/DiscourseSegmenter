#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Module providing parsing routines based on finite-state mechanisms.

Constants:

Methods:
constraint - function decorator making function execution safe against internal failures

Classes:
Tree - proxy tree class used for constructing segment nodes
SymTab
MatchProxy
FiniteStateParser

Exceptions:
Overflow - exception raised by SymTab class representing buffer overflow

@author = Jean VanCoppenolle, Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <vancoppenolle at uni dash potsdam dot de>, <sidarenk at uni dash potsdam dot de>

"""

##################################################################
# Libraries
from collections import defaultdict
from functools import wraps
from itertools import islice
from operator import itemgetter

import re
import sys
import warnings

##################################################################
# Methods
def constraint(func):
    """
    Function decorator making function execution safe against internal failures.

    @param func - reference to function which should be decorated

    @return self
    """
    @wraps(func)
    def decorate(match):
        return func(match)
        try:
            return func(match)
        except Exception as exc:
            warnings.warn('Exception in constraint: {0}'.format(exc))
        return False
    return decorate

##################################################################
# Classes
class Tree(object):
    '''Tree data structure'''
    def __init__(self, label, children=None, feats=None):
        self.label = label
        self._children = list(children or [])
        self.feats = feats or {}

    def __len__(self):
        return len(self._children)

    def __getitem__(self, index):
        return self.child(index)

    def __iter__(self):
        for child in self._children:
            yield child

    def set(self, feat, val):
        self.feats[feat] = val
        return self

    def get(self, feat):
        return self.feats.get(feat)

    def iter(self, label=None):
        for child in self:
            if label is None or (hasattr(child, 'label') and
                                 child.label == label):
                yield child
            if hasattr(child, 'iter'):
                for child_ in child.iter():
                    if label is None or (hasattr(child_, 'label') and
                                         child_.label == label):
                        yield child_

    def append(self, child):
        self._children.append(child)
        return self

    def extend(self, children):
        self._children.extend(children)
        return self

    def insert(self, index, child):
        self._children.insert(index, child)
        return self

    def remove(self, child):
        try:
            self._children.remove(child)
        except ValueError:
            pass
        return self

    def replace(self, node, *children):
        try:
            index = self._children.index(node)
        except ValueError:
            raise ValueError('node is not in tree')
        self.remove(node)
        for (offset, child) in enumerate(children):
            self._children.insert(index + offset, child)
        return self

    @property
    def first_child(self):
        return self.child(0)

    @property
    def last_child(self):
        return self.child(-1)

    def child(self, n):
        try:
            return self._children[n]
        except IndexError:
            pass

    def children(self, n):
        return self._children[:n]

    @property
    def first_terminal(self):
        for terminal in self.iter_terminals():
            return terminal

    @property
    def last_terminal(self):
        terminal = None
        for terminal in self.iter_terminals():
            pass
        return terminal

    def terminal(self, n):
        return next(islice(self.iter_terminals(), n, None))

    def terminals(self, n):
        return islice(self.iter_terminals(), None, n)

    @property
    def is_preterminal(self):
        return not any(isinstance(child, Tree) for child in self)

    def iter_preterminals(self):
        for child in self:
            if hasattr(child, 'is_preterminal'):
                if child.is_preterminal:
                    yield child
                else:
                    for preterminal in child.iter_preterminals():
                        yield preterminal

    def iter_terminals(self):
        for child in self:
            if not isinstance(child, Tree):
                yield child
            else:
                for terminal in child.iter_terminals():
                    yield terminal

    def pretty_print(self, stream=sys.stdout, depth=0, indent='    ',
                     term_print=lambda term: u'{form}/{pos}'.format(form=term['form'], \
                                                                        pos=term['pos']),
                     feat_print=lambda feat: u'{0}={1}'.format(feat[0], feat[1]),
                     encoding='utf-8'):
        emit = lambda out: stream.write('{0}{1}'.format(indent * depth, out))
        if self.feats:
            feat_str = ','.join(feat_print(item)
                                for item in self.feats.iteritems())
            emit('({0} [{1}]\n'.format(self.label, feat_str.encode(encoding)))
        else:
            emit('({0}\n'.format(self.label))
        for child in self:
            if hasattr(child, 'pretty_print'):
                child.pretty_print(stream=stream, depth=depth + 1,
                                   indent=indent, term_print=term_print,
                                   feat_print=feat_print, encoding='utf-8')
            else:
                emit('{0}{1}\n'.format(indent,
                                       term_print(child).encode(encoding)))
        emit(')\n')

    def __str__(self):
        """Return string representation of tree."""
        ostring = u""
        if self.feats:
            feat_str = u','.join([u"{0}={1}".format(item[0], item[1])
                                 for item in self.feats.iteritems()])
            ostring = u"({0} [{1}]\n".format(self.label, feat_str)
        else:
            ostring = self.label + '\n'

        for child in self:
            if isinstance(child, Tree):
                ostring += str(child) + '\n'
            else:
                ostring += u'{form}/{pos}'.format(form=child['form'], pos=child['pos'])
        ostring += ")\n"
        return ostring


class SymTab(object):
    class Overflow(Exception):
        pass

    _CAPACITY = 255

    def __init__(self):
        self._str_sym_map = {}
        self._sym_str_map = {}

    def encode(self, string):
        try:
            sym = self._str_sym_map[string]
        except KeyError:
            idx = len(self._str_sym_map)
            if idx > self._CAPACITY:
                raise self.Overflow
            sym = hex(idx)[2:].zfill(2).decode('hex')
            self._str_sym_map[string] = sym
            self._sym_str_map[sym] = string
        return sym

    def decode(self, sym):
        return self._sym_str_map.get(sym)


class MatchProxy(object):
    def __init__(self, match, nodes):
        self._match = match
        self._nodes = nodes

    def __getitem__(self, group):
        start, end = self._match.start(group), self._match.end(group)
        return self._nodes[start:end]

class FiniteStateParser(object):
    _RE_VAR = re.compile('(%[^%]+%)', re.I)
    _RE_CAT = re.compile('(?<!P)<(?!=)([^>]+)>', re.I)

    def __init__(self, root_cat='ROOT'):
        self.root_cat = root_cat
        self.cats = set()
        self._sym_tab = SymTab()
        self._vars = {}
        self._rules = defaultdict(lambda: defaultdict(list))
        self._regexps = None

    @classmethod
    def from_file(cls, path):
        parser = cls()
        with open(path) as fp:
            for line_no, line in enumerate(fp):
                comment_pos = line.find('#')
                if comment_pos > -1:
                    line = line[:comment_pos]
                line = line.strip()
                if not line:
                    continue
                elif '=' in line:
                    try:
                        name, pattern = line.split('=', 1)
                    except ValueError:
                        raise SyntaxError(
                            'malformed variable definition (line {0})'.format(
                                line_no))
                    parser.define(name.strip(), pattern.strip())
                else:
                    try:
                        (level, lhs, rhs) = line.split('\t')
                    except ValueError:
                        raise SyntaxError('malformed rule (line {0})'.format(
                            line_no))
                    parser.add_rule(lhs, rhs, level=level)
        return parser

    def define(self, name, pattern):
        if name in self._vars:
            raise ValueError('{0} already defined'.format(name))
        self._vars[name] = self._replace_vars(pattern)
        return self

    def add_rule(self, lhs, rhs, level=0, constraint=None, group=0,
                 feats=None):
        if level < 0:
            raise ValueError('level must be a positive number')
        elif group < 0:
            raise ValueError('group must be a positive number')
        self._rules[level][lhs].append({'regex': self._compile_rhs(rhs),
                                        'constraint': constraint,
                                        'group': group,
                                        'feats': feats})
        return self

    def parse(self, tokens, catgetter=lambda tok: tok):
        nodes = tokens
        for lvl, rules in sorted(self._rules.iteritems(), key = itemgetter(0)):
            # print >> sys.stderr, "parser.parse: lvl =", repr(lvl)
            nodes = self._parse_level(rules, nodes, catgetter)
            # print >> sys.stderr, "parser.parse: nodes =", repr(nodes)
        return Tree(self.root_cat, nodes)

    def _parse_level(self, rules, nodes, catgetter):
        tag_string = self._make_tag_string(nodes, catgetter)
        tree = None
        # print >> sys.stderr, "1) tag_string =", repr(tag_string)
        for lhs, rhs_specs in rules.iteritems():
            # print >> sys.stderr, "lhs = ", repr(lhs)
            for spec in rhs_specs:
                pos = 0
                next_nodes = []
                append = next_nodes.append
                extend = next_nodes.extend
                # print >> sys.stderr, "regex = ", repr(spec['regex'].pattern)
                for match in spec['regex'].finditer(tag_string):
                    # print >> sys.stderr, "regex matched"
                    group = spec['group']
                    start = match.start(group)
                    end = match.end(group)
                    proxy = MatchProxy(match, nodes)
                    constraint = spec['constraint']
                    if constraint is not None:
                        try:
                            flag = constraint(proxy)
                        except Exception as exc:
                            flag = False
                            warnings.warn('Exception in constraint: {0}'.format(lhs, exc))
                        if not flag:
                            continue
                    feats = spec['feats']
                    if feats is not None:
                        try:
                            feats = feats(proxy)
                        except Exception as exc:
                            warnings.warn(
                                'Exception in feature: {0}'.format(lhs, exc))
                    extend(nodes[idx] for idx in xrange(pos, start))
                    pos = end
                    tree = Tree(lhs, nodes[start:end], feats=feats)
                    # print >> sys.stderr, "_parse_level: tree =", repr(tree.pretty_print())
                    append(tree)
                # print >> sys.stderr, "next_nodes =", repr(next_nodes)
                if not next_nodes:
                    continue
                extend(nodes[idx] for idx in xrange(pos, len(nodes)))
                nodes = next_nodes
                tag_string = self._make_tag_string(nodes, catgetter)
                # print >> sys.stderr, "2) tag_string =", repr(tag_string)
        return nodes

    def _make_tag_string(self, nodes, catgetter):
        return ''.join(self._encode(node.label)
                       if isinstance(node, Tree)
                       else self._encode(catgetter(node))
                       for node in nodes)

    def _replace_vars(self, pattern):
        return self._RE_VAR.sub(lambda match: self._vars[match.group(1)[1:-1]],
                                              pattern)

    def _compile_rhs(self, rhs):
        rhs = self._replace_vars(rhs)
        clean_rhs = []
        for line in rhs.splitlines():
            comment_pos = line.find('#')
            if comment_pos != -1:
                line = line[:comment_pos]
            clean_rhs.append(line.strip().replace(' ', ''))
        rhs = ''.join(clean_rhs)
        rhs = self._RE_CAT.sub(lambda match: self._encode(match.group(1),
                                                          escape=True),
                               rhs)
        return re.compile(rhs)

    def _encode(self, cat, escape=False):
        sym = self._sym_tab.encode('<{0}>'.format(cat))
        if escape:
            sym = re.escape(sym)
        return sym

    def _decode(self, sym):
        return self._sym_tab.decode(sym)
