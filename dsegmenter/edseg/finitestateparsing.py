#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

"""Module providing parsing routines based on finite-state mechanisms.

Constants:

Methods:

constraint - function decorator making function execution safe against internal
failures

Classes:
Tree - auxiliary tree class used for constructing segment nodes
SymTab - auxiliary class mapping strings to hex codes
MatchProxy - interface class allowing to address matched discourse nodes
FiniteStateParser - match engine for matching rules

Exceptions:
Overflow - exception raised by SymTab class when buffer overflows

@author = Jean VanCoppenolle, Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <vancoppenolle at uni dash potsdam dot de>,
<sidarenk at uni dash potsdam dot de>

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
    """Decorator making function execution safe against internal failures.

    Args:
    func (method):
    reference to function which should be decorated

    Returns:
    self

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
    """
    Auxiliary tree class used for constructing segment nodes

    Instance variables:
    feats - dictionary of node features and their values
    label - label of the root node
    _children - list of node's children

    Property methods:
    is_preterminal - predicate checking if any of node's children are terminals

    Public methods:
    get - obtain feature value
    set - set feature value
    iter - iterate over node's children
    append - add new child to the list of children
    extend - form new list of children by joining it with another list
    insert - add new child to the list of children at specific position
    remove - remove given child from child list
    replace - replace given child node with a list of other children
    first_child - return first child
    last_child - return last child
    child - return n-th child
    children - return list of n first children
    first_terminal - return first terminal node in the list of children
    last_terminal - return last terminal node in the list of children
    terminal - return n-th terminal node
    terminals - return list of n terminals
    iter_terminals - iterate over terminal nodes
    iter_preterminals - iterate over preterminal nodes
    pretty_print - output nice string representation of the tree
    """

    def __init__(self, label, children=None, feats=None):
        """
        Class constructor

        @param label - string label for the root node
        @param children - list of node's children
        @param feats - dictionary of node's features
        """
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
        """
        Set feature value

        @param feat - name of the feature whose value should be updated
        @param val - new feature's value

        @return self
        """
        self.feats[feat] = val
        return self

    def get(self, feat):
        """
        Obtain feature value

        @param feat - name of the feature whose value should be retrieved

        @return feature value
        """
        return self.feats.get(feat)

    def iter(self, label=None):
        """
        Iterate over node's children

        @param label - label of node's which should be considered
                       during iteration (use None for all children)
        """
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
        """
        Add new child to the list of node's children

        @param child - node to be added

        @return self
        """
        self._children.append(child)
        return self

    def extend(self, children):
        """
        Add new children to the current list of node's children

        @param children - list of children to be joined with the children
                          of current node

        @return self
        """
        self._children.extend(children)
        return self

    def insert(self, index, child):
        """
        Add new child to the list of children at specific position

        @param index - insertion index for the new node
        @param child - node to be added

        @return self
        """
        self._children.insert(index, child)
        return self

    def remove(self, child):
        """
        Remove given child from the list of children

        @param child - node to be removed

        @return self
        """
        try:
            self._children.remove(child)
        except ValueError:
            pass
        return self

    def replace(self, node, *children):
        """
        Replace given child node with a list of other children

        @param node - node to be replaced
        @param children - list of replacement nodes

        @return self
        """
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
        """
        Retrieve first child of the current node

        @return reference to the first child of the current node
        """
        return self.child(0)

    @property
    def last_child(self):
        """
        Retrieve last child of the current node

        @return reference to the last child of the current node
        """
        return self.child(-1)

    def child(self, n):
        """
        Retrieve n-th child of the current node

        @param n - index of the child to be retrieved

        @return reference to the n-th child of the current node
        """
        try:
            return self._children[n]
        except IndexError:
            pass

    def children(self, n):
        """
        Retrieve list of n first children

        @param n - number of children to be retrieved

        @return list of n first children
        """
        return self._children[:n]

    @property
    def first_terminal(self):
        """
        Retrieve first terminal node in the list of children

        @return reference to first descendant terminal node
        """
        for terminal in self.iter_terminals():
            return terminal

    @property
    def last_terminal(self):
        """
        Retrieve last terminal node in the list of children

        @return reference to last descendant terminal node
        """
        terminal = None
        for terminal in self.iter_terminals():
            pass
        return terminal

    def terminal(self, n):
        """
        Retrieve last terminal node in the list of children

        @param n - index of the terminal node

        @return reference to last descendant terminal node
        """
        return next(islice(self.iter_terminals(), n, None))

    def terminals(self, n):
        """
        Retrieve list of descendant terminal nodes

        @param n - number of terminal nodes to retrieve

        @return list of n first descendant terminals
        """
        return islice(self.iter_terminals(), None, n)

    @property
    def is_preterminal(self):
        """Check if all of node's children are terminals

        Returns:
        (bool):
        true if all of node's children are terminals, false otherwise

        """
        return not any(isinstance(child, Tree) for child in self)

    def iter_preterminals(self):
        """
        Iterate over descendant preterminal nodes

        @return iterator over pre-terminals
        """
        for child in self:
            if hasattr(child, 'is_preterminal'):
                if child.is_preterminal:
                    yield child
                else:
                    for preterminal in child.iter_preterminals():
                        yield preterminal

    def iter_terminals(self):
        """
        Iterate over descendant terminal nodes

        @return iterator over terminals
        """
        for child in self:
            if not isinstance(child, Tree):
                yield child
            else:
                for terminal in child.iter_terminals():
                    yield terminal

    def pretty_print(self, a_stream=sys.stdout, a_depth=0, a_indent='    ',
                     a_term_print=lambda term: u'{form}/{pos}'.format(
                         form=term['form'], pos=term['pos']),
                     a_feat_print=lambda feat: u'{0}={1}'.format(
                         feat[0], feat[1]),
                     a_encoding='utf-8'):
        """
        Output nice string representation of the current tree

        @param a_stream - output stream to be used
        @param a_depth - nestedness level of the current tree from the root
        @param a_indent - checrater to be used for indentation
        @param a_term_print - cutom function for outputting terminals
        @param a_feat_print - cutom function for outputting features
        @param a_encoding - encoding for the final output string

        @return \c void
        """
        emit = lambda out: a_stream.write('{0}{1}'.format(
            a_indent * a_depth, out))
        if self.feats:
            feat_str = ','.join(a_feat_print(item)
                                for item in self.feats.iteritems())
            emit('({0} [{1}]\n'.format(self.label,
                                       feat_str.encode(a_encoding)))
        else:
            emit('({0}\n'.format(self.label))
        for child in self:
            if hasattr(child, 'pretty_print'):
                child.pretty_print(a_stream=a_stream, a_depth=a_depth + 1,
                                   a_indent=a_indent,
                                   a_term_print=a_term_print,
                                   a_feat_print=a_feat_print,
                                   a_encoding=a_encoding)
            else:
                emit('{0}{1}\n'.format(a_indent,
                                       a_term_print(child).encode(a_encoding)))
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
                ostring += u'{form}/{pos}'.format(
                    form=child['form'], pos=child['pos'])
        ostring += ")\n"
        return ostring


class SymTab(object):
    """
    Auxiliary class mapping strings to hex codes

    Constants:
    _CAPACITY - maximum allowed number od different string to be mapped

    Instance variables:
    _str_sym_map - internal mapping from string to hexadecimal codes
    _sym_str_map - internal mapping from hexadecimal codes to strings

    Public methods:
    encode - map string to hexadecimal code
    decode - map hexadecimal code to string

    Exceptions:
    Overflow - exception raised when the number of mapped strings exceeds limit
    """

    class Overflow(Exception):
        """Custom exception.

        Thrown when the number of mapped strings exceeds capacity

        This class subclasses `Exception`

        """

        pass

    _CAPACITY = 255

    def __init__(self):
        """
        Class constructor
        """
        self._str_sym_map = {}
        self._sym_str_map = {}

    def encode(self, string):
        """
        Convert string to hexadecimal code

        @param string - string to be converted

        @return hexadecimal code corresponding to that string
        """
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
        """
        Convert hexadecimal code to string

        @param sym - string to be converted

        @return hexadecimal code corresponding to that string
        """
        return self._sym_str_map.get(sym)


class MatchProxy(object):
    """
    Interface class allowing to address matched discourse nodes

    Instance variables:
    _match - reference to rule match
    _nodes - reference to the list of nodes to which match was applied
    """

    def __init__(self, match, nodes):
        """Class constructor

        @param match - reference to rule match
        @param nodes - reference to the list of nodes to which match was
        applied

        """
        self._match = match
        self._nodes = nodes

    def __getitem__(self, group):
        """
        Return nodes corresponding to the n-th matched group

        @param group - number of match group to return

        @return nodes corresponding to the n-th matched group
        """
        start, end = self._match.start(group), self._match.end(group)
        return self._nodes[start:end]


class FiniteStateParser(object):
    """Match engine used for matching rules

    Class methods:
    from_file - create an automaton instance from file with rules

    Constants:
    _RE_VAR - regexp intended to match macros in rules
    _RE_CAT - regexp intended to match non-terminal nodes in rules

    Instance variables:
    root_cat - label to be used for the root node of the constructed tree
    cats - set of possible tree labels
    _sym_tag - internal mapping of symbolic strings to hex codes
    _vars - internal mapping from macro names to their values
    _rules - internal mapping from rule levels to compiled rules

    Public methods:
    add_rule - convert symbolic rule to a regular expression and add it to
    common automaton
    define - add macro definition to the current set of rules
    parse - parse given rules instance and add it to the rule cascade

    """

    _RE_VAR = re.compile('(%[^%]+%)', re.I)
    _RE_CAT = re.compile('(?<!P)<(?!=)([^>]+)>', re.I)

    def __init__(self, root_cat='ROOT'):
        """Class constructor

        @param root_cat - label to be used for the root node of the constructed
        tree

        """
        self.root_cat = root_cat
        self.cats = set()
        self._sym_tab = SymTab()
        self._vars = {}
        self._rules = defaultdict(lambda: defaultdict(list))

    @classmethod
    def from_file(cls, path):
        """
        Create an automaton instance from file with rules

        @param cls - reference to the current class
        @param path - path to the file containing parser rules
        """
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
        """
        Add macro definition to the current set of rules

        @param name - name of the macro
        @param pattern - macro replacement

        @return self
        """
        if name in self._vars:
            raise ValueError('{0} already defined'.format(name))
        self._vars[name] = self._replace_vars(pattern)
        return self

    def add_rule(self, lhs, rhs, level=0, constraint=None, group=0,
                 feats=None):
        """Convert rule to regexp and add it to common automaton

        @param lhs - left hand side of the rule (is used for matching)
        @param rhs - left hand side of the rule (is used as replacement)
        @param level - precedence level of the rule
        @param constraint - additional check to be applied on the rule match
        @param group - regexp group to which constraint check should be applied
        @param feats - custom function for extracting features from nodes

        @return self

        """
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
        """
        Parse given rules instance and add it to the rule cascade

        @param tokens - tokens to which rule matching shoudl be applied
        @param catgetter - custom function for obtaining tokens category

        @return newly constructed segment tree
        """
        nodes = tokens
        for lvl, rules in sorted(self._rules.iteritems(), key=itemgetter(0)):
            nodes = self._parse_level(rules, nodes, catgetter)
        return Tree(self.root_cat, nodes)

    def _parse_level(self, rules, nodes, catgetter):
        tag_string = self._make_tag_string(nodes, catgetter)
        tree = None
        for lhs, rhs_specs in rules.iteritems():
            for spec in rhs_specs:
                pos = 0
                next_nodes = []
                append = next_nodes.append
                extend = next_nodes.extend
                for match in spec['regex'].finditer(tag_string):
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
                            warnings.warn('Exception in constraint:'
                                          ' {0}'.format(lhs, exc))
                            raise
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
                    append(tree)
                if not next_nodes:
                    continue
                extend(nodes[idx] for idx in xrange(pos, len(nodes)))
                nodes = next_nodes
                tag_string = self._make_tag_string(nodes, catgetter)
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
