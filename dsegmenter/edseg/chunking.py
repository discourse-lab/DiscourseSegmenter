#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""
Module providing internal rule-based clause segmenter

Constants:

Methods:
catgetter - method for obtaining category of token

Classes:
FeatureMatrix - class mapping CONLL features to bit matrix
ClauseSegmenter - class for doing clause segmentation

Exceptions:
UnificationFailure - exception raise on non-merged feature bits

@author = Jean VanCoppenolle, Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <vancoppenolle at uni dash potsdam dot de>,
        <sidarenk at uni dash potsdam dot de>

"""

##################################################################
# Libraries
from __future__ import unicode_literals

from .finitestateparsing import constraint, FiniteStateParser

from copy import deepcopy
import sys


##################################################################
# Methods
def catgetter(token):
    return token["pos"]


##################################################################
# Exceptions
class UnificationFailure(Exception):
    pass


##################################################################
class FeatureMatrix(object):
    """Class for converting CONLL features to bit matrices.

    Class constants:
    FEATS - nominal names of the features
    _FEAT_INDICES - indices of the features in the bit matrix

    Instance variables:
    _bits - internal bit matrix of features

    Public methods:
    from_string - initiate feature matrix from string representation
    from_dict - initiate feature matrix from dictionary of feature names and
                values
    unify - make an intersection of features in the current matrix with the
            features from another instance
    unifies - check if the intersection of the current feature matrix with
              the matrix from another instance is not empty

    """

    FEATS = [
        "nom",
        "acc",
        "dat",
        "gen",
        "sg",
        "pl",
        "masc",
        "fem",
        "neut",
    ]

    _FEAT_INDICES = dict((feat, idx) for (idx, feat) in enumerate(FEATS))

    def __init__(self, feats):
        """
        Class constructor.

        @param feats - list of feature names
        """
        bits = 0
        for feat in feats:
            idx = self._FEAT_INDICES.get(feat.lower())
            if idx is not None:
                bits |= 1 << idx
        if not bits & 0xf:
            bits |= 0xf
        if not (bits >> 4) & 0x3:
            bits |= 0x3 << 4
        if not (bits >> 6) & 0x7:
            bits |= 0x7 << 6
        self._bits = bits

    @classmethod
    def from_string(cls, feat_str):
        """
        Initiate feature matrix from string representation

        @param feat_str - string containing feature names and their values

        @return FeatureMatrix instance
        """
        return cls([feat.strip() for feat in feat_str.split('|')])

    @classmethod
    def from_dict(cls, feat_dict):
        """
        Initiate feature matrix from string representation

        @param feat_dict - dictionary containing feature names and their values

        @return FeatureMatrix instance
        """
        return cls([v for v in feat_dict.itervalues()])

    def unify(self, other):
        """Intersect current features with the features from another instance

        @param other - another FeatureMatrix instance

        @return this FeatureMatrix instance

        """
        if not hasattr(other, "_bits"):
            return False
        bits = self._bits & other._bits
        if not self._unified(bits):
            raise UnificationFailure
        self._bits = bits
        return self

    def unifies(self, other):
        """Check if intersection with another instance is not empty

        @param other - another FeatureMatrix instance

        @return this FeatureMatrix instance

        """
        if not hasattr(other, "_bits"):
            return False
        return self._unified(self._bits & other._bits)

    def _unified(self, bits):
        return (bits & 0xf) and (bits >> 4) & 0x3 and (bits >> 6) & 0x7

    def __str__(self):
        return bin(self._bits)[2:]


##################################################################
class Chunker(object):
    """
    Class for doing clause segmentation.

    Instance variables:
    _parser - internal finite-state parser for doing segmentation

    Public methods:
    chunk - perform clause chunking of the CONLL tree
    """

    def __init__(self):
        """
        Class constructor
        """
        self._parser = FiniteStateParser()
        self._setup_parser()

    def chunk(self, sent):
        """
        Perform clause chunking of the CONLL tree

        @param sent - CONLL sentence to parse

        @return list of clause segments
        """
        # convert word features to feature matrices
        # make a deep copy of sentence, in order not to use it destructively
        isent = deepcopy(sent)
        for token in isent:
            if token["pos"] in ("ART", "NE", "NN"):
                if isinstance(token["feat"], basestring):
                    token["feat"] = FeatureMatrix.from_string(token["feat"])
                elif isinstance(token["feat"], dict):
                    token["feat"] = FeatureMatrix.from_dict(token["feat"])
        return self._parser.parse(isent, catgetter=catgetter)

    def _setup_parser(self):
        add_rule = self._parser.add_rule

        add_rule("NC",
                 """
                 <PPER>
                 """,
                 level=1)

        @constraint
        def nc_month_spec_constraint(match):
            if match[2][0]["lemma"] not in ("Anfang", "Mitte", "Ende"):
                return False
            return match[3][0]["lemma"] in ("Januar",
                                            "Februar",
                                            "März",
                                            "Maerz",
                                            "April",
                                            "Mai",
                                            "Juni",
                                            "Juli",
                                            "August",
                                            "September",
                                            "Oktober",
                                            "November",
                                            "Dezember")

        add_rule("NC",
                 """
                 (?:
                 ^
                 |
                 [^<ART><CARD><PDAT><PDS><PIAT><PPOSAT>]
                 )
                 (
                 (<NN>)
                 (<NN>)
                 )
                 """,
                 constraint=nc_month_spec_constraint,
                 group=1, level=1)

        @constraint
        def nc_det_noun_agreement(match):
            det = match[1]
            if not det:
                return True
            noun = match[2][0]
            try:
                if hasattr(noun["feat"], "unify"):
                    noun["feat"].unify(det[0])
                else:
                    return False
            except UnificationFailure:
                return False
            return True

        add_rule("NC",
                 """
                 (?:
                 (<ART>)<PIAT>?
                 |
                 [<CARD><PDAT><PDS><PIAT><PPOSAT>]
                 )?
                 (?:
                 (?:
                 <ADJA><ADJA>?
                 )
                 (?:
                 <$,>
                 (?:
                 <ADJA><ADJA>?
                 )
                )*
                 )?
                 ([<NE><NN>])
                 """,
                 constraint=nc_det_noun_agreement,
                 level=1)

        add_rule("NC",
                 """
                 (?:
                 <ART><PIAT>?
                 |
                 [<PDAT><PDS><PPOSAT><PIAT><CARD>]
                 )
                 <PC>
                 <NC>
                 """,
                 level=3)

        add_rule("PC",
                 """
                 <APPR>   # preposition
                 <NC>?
                 <PDS>
                 """,
                 level=2)

        @constraint
        def pc_genitive_adjunct_constraint(match):
            node = match[1][0]
            if node.last_child["pos"] != "NN":
                return False
            art = node.first_child
            if art is None or art["pos"] != "ART":
                return False
            if "feat" not in art or not hasattr(art["feat"], "unifies"):
                return False
            return art["feat"].unifies(FeatureMatrix("gen"))

        add_rule("PC",
                 """
                 [<APPR><APRRART>]
                 <NC>
                 (?:
                 <KON>
                 <NC>
                 )*
                 (<NC>)
                 (?:
                 <PDS>
                 |
                 [<APZR><PROAV>]
                 )?
                 """,
                 constraint=pc_genitive_adjunct_constraint,
                 level=2)

        add_rule("PC",
                 """
                 [<APPR><APPRART>]   # preposition
                 <APPR>?             # ("bis an das Ende")
                 (?:
                 <ADV>           # adverbial chunk ("von damals")
                 (?:             # optional conjunction
                 <KON>
                 <ADV>
                 )?
                 |
                 <CARD>          # cardinal ("bis 1986")
                 (?:             # optional conjunction
                 <KON>
                 <CARD>
                 )?
                 |
                 <NC>            # noun chunk
                 (?:             # optional conjunction
                 <KON>
                 <NC>
                 )?
                 )
                 [<APZR><PROAV>]?    # optional pronominal adverb
                 """,
                 level=2)

        add_rule("AC",
                 """
                 <ADV>*
                 <PTKNEG>?
                 <ADJD>+
                 """,
                 level=3)

        add_rule("AC",
                 """
                 <PTKA>?
                 <ADV>+
                 """,
                 level=3)

        add_rule("AC",
                 """
                 <PROAV>
                 """,
                 level=3)
