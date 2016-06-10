#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module providing rule-based clause segmenter

.. moduelauthor:: Jean VanCoppenolle

"""

##################################################################
# Libraries
from __future__ import absolute_import

from dsegmenter.edseg.chunking import Chunker
from dsegmenter.edseg.finitestateparsing import FiniteStateParser, Tree
from dsegmenter.edseg.util import match as match_
from dsegmenter.edseg.data import DELIMS, DELIM_NAMES, finite_verbs

import sys


##################################################################
# Methods
def catgetter(node):
    """
    Method for obtaining category of CONLL node.

    @param node - CONLL node

    @return node's category
    """
    if hasattr(node, 'label'):
        return node.label
    form = unicode(node['form'])
    if form in DELIM_NAMES:
        return DELIM_NAMES[form]
    return node['pos']


##################################################################
# Class
class ClauseSegmenter(object):
    """Class for perfoming discourse segmentation on CONLL dependency trees.

    Attributes:
      _chunker: internal rule-based clause chunker
      _parser: internal finite-state parser

    """

    def __init__(self, **kwargs):
        """Class constructor.

        @param a_chunker - clause chunker to use

        """
        chunker = kwargs.get('chunker')
        if chunker is None:
            self._chunker = Chunker()
        else:
            self._chunker = chunker
        self._parser = FiniteStateParser()
        self._setup_parser()

    def segment(self, sent):
        """Method for segmenting CONLL trees.

        @param sent - CONLL tree to process

        @return sentence-level discourse segment

        """
        self._prepare_tokens(sent)
        chunk_tree = self._chunker.chunk(sent)
        tree = self._parser.parse(chunk_tree, catgetter=catgetter)
        self._flatten(tree, ('AC', 'NC', 'FVG', 'IVG'))
        return tree

    def _prepare_tokens(self, sent):
        for token in sent:
            verb_type = finite_verbs.get(token['form'], default=None)
            if verb_type is not None:
                token['pos'] = 'V{0}FIN'.format(verb_type)

    def _flatten(self, node, labels, parent=None):
        if not isinstance(node, Tree):
            return
        for child in node:
            self._flatten(child, labels, parent=node)
        if node.label in labels:
            parent.replace(node, *node)

    def _setup_parser(self):
        define = self._parser.define
        add_rule = self._parser.add_rule

        ##########
        # Macros #
        ##########

        define('VFIN',
               '''
               <VAFIN>
               <VMFIN>
               <VVFIN>
               ''')

        define('VINF',
               '''
               <VAINF>
               <VAPP>
               <VMINF>
               <VMPP>
               <VVINF>
               <VVIZU>
               <VVPP>
               ''')

        define('V',
               '''
               <VAFIN>
               <VMFIN>
               <VVFIN>
               <VAINF>
               <VAPP>
               <VMINF>
               <VMPP>
               <VVINF>
               <VVIZU>
               <VVPP>
               ''')

        define('PUNCT',
               '''
               <$,>
               <$(>
               <$.>
               ''')

        define('EOS',
               '''
               <$.>
               <$,>
               ''')

        define('DASH',
               '''
               <EM_DASH>
               <EN_DASH>
               <FIGURE_DASH>
               <HORIZONTAL_BAR>
               ''')

        define('VG',
               '''
               <FVG>
               <IVG>
               ''')

        define('CLAUSE',
               '''
               <RelCl>
               <InfCl>
               <IntCl>
               <SntSubCl>
               <InfSubCl>
               <MainCl>
               ''')

        define('BASIC_CONTENT',
               '''
               (?:
               [^%PUNCT%<KON>%VG%]
               (?:
               <KON>?
               [^%PUNCT%<KON>%VG%]
               )?
               )*
               ''')

        define('CONTENT',
               '''
               (?:
               [^%PUNCT%<KON>%VG%]
               (?:
               [<$,><KON>]
               [^%PUNCT%<KON>%VG%]
               |
               [%CLAUSE%]
               <$,>
               )?
               )*
               ''')

        define('BASIC_TRAILER',
               '''
               (?:
               [<APPR><APPRART><KOKOM>]
               [^%PUNCT%<KON>%VG%]+
               |
               <PC>
               )
               ''')

        ##########################
        # Parenthesized segments #
        ##########################

        for ldelim, rdelim in DELIMS.iteritems():
            ldelim_name = DELIM_NAMES[ldelim]
            rdelim_name = DELIM_NAMES[rdelim]
            add_rule('Paren', '<{0}>[^<{1}>]+<{1}>'.format(ldelim_name,
                                                           rdelim_name))

        ###############
        # Verb groups #
        ###############

        def get_verb(match, group=0):
            main, modal, aux = None, None, None
            for token in match[group]:
                pos = token['pos']
                if pos.startswith('VV'):
                    main = token
                elif pos.startswith('VM'):
                    modal = token
                elif pos.startswith('VA'):
                    aux = token
            if main:
                return main
            elif modal:
                return modal
            return aux

        add_rule('FVG',
                 '''
                 (
                 <PTKZU>?
                 [%VINF%]+
                 [%VFIN%]
                 |
                 # gehen !!! added by W. Sidorenko (remove if it causes errors)
                 # lassen or simply `gehen' in case of tagging mistakes
                 <VVINF>
                 <VVINF>
                 )
                 (?:
                 [^<NC><PC>]
                 |
                 $
                 )
                 ''',
                 group=1,
                 feats=lambda match: {'verb': get_verb(match, group=1)},
                 level=5)

        add_rule('FVG',
                 '''
                 (?:
                 <VVPP>         # ausgenommen
                 <VAINF>        # werden
                 [%VFIN%]       # soll
                 |
                 [%VFIN%]       # soll
                 <VVPP>         # ausgenommen
                 <VAINF>        # werden
                 |
                 [%VFIN%]
                 [%VINF%]*
                 )
                 ''',
                 feats=lambda match: {'verb': get_verb(match)},
                 level=5)

        add_rule('IVG',
                 '''
                 [%VINF%]*
                 <PTKZU>?
                 [%VINF%]+
                 ''',
                 feats=lambda match: {'verb': get_verb(match)},
                 level=5)

        ################################
        # Basic clauses (no embedding) #
        ################################

        add_rule('RelCl',
                 '''
                 <APPR>?                 # optional preposition
                 [<PRELS><PRELAT>]       # relative pronoun
                 %BASIC_CONTENT%         # clause content
                 (
                 # verb group (error tolerance: should actually be finite)
                 [%VG%]
                 )
                 %BASIC_TRAILER%?        # optional trailer
                 # optional end of sentence punctuation
                 [%EOS%]?
                 ''',
                 feats=lambda match: {'verb': match[1][0].get('verb')},
                 level=6)

        add_rule('RelCl',
                 '''
                 <KON>                   # conjunction
                 (
                 <APPR>?             # optional preposition
                 [<PRELS><PRELAT>]   # relative pronoun
                 %BASIC_CONTENT%     # clause content
                 (
                 # verb group (error tolerance: should actually be finite)
                 [%VG%]
                 )
                 %BASIC_TRAILER%?    # optional trailer
                 [%EOS%]?              # optional end of sentence punctuation
                 )
                 ''',
                 group=1,
                 feats=lambda match: {'verb': match[2][0].get('verb')},
                 level=7)

        add_rule('RelCl',
                 '''
                 <RelCl>                 # relative clause
                 <KON>                   # conjunction
                 (
                 %BASIC_CONTENT%     # clause content
                 (
                 # verb group (error tolerance: should actually be finite)
                 [%VG%]
                 )
                 %BASIC_TRAILER%?    # optional trailer
                 [%EOS%]?              # optional end of sentence punctuation
                 )
                 ''',
                 group=1,
                 feats=lambda match: {'verb': match[2][0].get('verb')},
                 level=7)

        def complex_that(match):
            tokens = list(match[1][0].iter_terminals())
            return (match_(tokens, ('Dadurch', u'Dafür', 'Dafuer')) or
                    match_(tokens, 'Aufgrund', 'dessen') or
                    match_(tokens, 'Auf', 'Grund', 'dessen'))

        add_rule('SntSubCl',
            '''
            ^                       # start of sentence
            (?:                       # cases like
                <AC>                # "Dadurch, dass"
            |
                <PC>                # or "Aufgrund dessen", dass
            )
            <$,>                    # comma
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                [%VG%]          # because of possible tagging errors, we don't require verb
                                # to be finite
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            constraint=complex_that,
            level=7)

        add_rule('SntSubCl',
            '''
            (?:
                ^                   # start of sentence
            |
                <$(>                # or dash
            )?
            [<AC><APPR>]?           # optional adverb or preposition ("außer wenn ...")
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                 [%VG%]          # because of possible tagging errors, we don't require verb
                                 # to be finite
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=7)

        add_rule('SntSubCl',
            '''
            <KON>                   # conjunction
            (
                <APPR>?             # optional preposition ("außer wenn ...")
                <KOUS>
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # because of possible tagging errors, we don't require verb
                                    # to be finite
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('SntSubCl',
            '''
            <SntSubCl>              # sentential subordinate clause
            <KON>                   # conjunction
            (
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # because of possible tagging errors, we don't require verb
                                    # to be finite
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]             # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=14)

        add_rule('InfSubCl',
            '''
            # (?:
            #     ^                   # start of sentence
            # |
            #     <$,>                # or comma
            # )
            [<AC><APPR>]?           # optional adverb or preposition ("außer um ...")
            <KOUI>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            (?: <$.> | <$,>)?       # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=7)

        add_rule('InfSubCl',
            '''
            <InfSubCl>              # non-finite subordinate clause
            <KON>                   # conjunction
            (
                %BASIC_CONTENT%     # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('InfSubCl',
            '''
            <KON>                   # conjunction
            (
                <KOUI>              # subordinating conjunction
                %BASIC_CONTENT%     # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        add_rule('IntCl',
            '''
                [<PWS><PWAT><PWAV>] # interrogative pronoun
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            ''',
                 feats = lambda match: {'verb': match[1][0].get('verb')},
                 level = 7)

        # # NOTE: apparently not needed any more
        # add_rule('IntCl',
        #     '''
        #     <KON>
        #     (
        #         [<PWS><PWAT><PWAV>] # interrogative pronoun
        #         %BASIC_CONTENT%     # clause content
        #         (
        #             [%VG%]          # verb group
        #         )
        #         %BASIC_TRAILER%?    # optional trailer
        #         <$.>?               # optional end of sentence punctuation
        #     )
        #     ''',
        #     group=1,
        #     feats=lambda match: {'verb': match[2][0].get('verb')},
        #     level=8)

        add_rule('IntCl',
            '''
            <IntCl>
            <KON>
            (
                %BASIC_CONTENT%     # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=8)

        ######################
        # Clause combination #
        ######################

        add_rule('IntCl',
            '''
            <IntCl>
            <RelCl>
            ''',
            level=9)

        add_rule('IntCl',
            '''
            <IntCl>
            (?:
                <KON>?
                <IntCl>
            )+
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <SntSubCl>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfSubCl>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            <Paren>
            ''',
            level=10)

        add_rule('RelCl',
            '''
            <RelCl>
            (?:
                <KON>?
                <RelCl>
            )+
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <RelCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <IntCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <InfSubCl>
            ''',
            level=11)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            (?:
                <KON>
                <SntSubCl>
            )+
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <RelCl>
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <IntCl>
            ''',
            level=11)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            (?:
                <KON>
                <InfSubCl>
            )+
            ''',
            level=11)

        ######################################
        # Basic clauses (embedding possible) #
        ######################################

        add_rule('RelCl',
            '''
            <$,>                    # comma
            <APPR>?                 # optional preposition
            [<PRELS><PRELAT>]       # relative pronoun
            %CONTENT%               # clause content
            (
                [%VG%]              # verb group (error tolerance: should actually be finite)
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('RelCl',
            '''
            <KON>                   # conjunction
            (
                <APPR>?             # optional preposition
                [<PRELS><PRELAT>]   # relative pronoun
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('RelCl',
            '''
            <RelCl>                 # relative clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group (error tolerance: should actually be finite)
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('SntSubCl',
            '''
            ^                       # start of sentence
            (                       # cases like
                <AC>                # "Dadurch, dass"
            |
                <PC>                # or "Aufgrund dessen", dass
            )
            <$,>                    # comma
            <KOUS>                  # subordinating conjunction
            %BASIC_CONTENT%         # clause content
            (
                  [%VG%]          # because of possible tagging errors, we don't require verb
                                  # to be finite
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            constraint=complex_that,
            level=12)

        add_rule('SntSubCl',
            '''
             ^                   # start of sentence
            (
                <FVG>            # finite verb
                %CONTENT%        # clause content
                (?: <IVG> )?
                <$,>
            )
            <AC>?
            <FVG>
            ''',
            group=1,
            level=12)

        add_rule('SntSubCl',
            '''
            # (?:
            #     ^                   # start of sentence
            # |
            #     <$,>                # or comma
            # )
            [<AC><APPR>]?           # optional adverb or preposition ("außer wenn ...")
            <KOUS>                  # subordinating conjunction
            %CONTENT%               # clause content
            (
                 [%VG%]          # because of possible tagging errors, we don't require verb
                                 # to be finite
                 |
                 <PTKVZ>
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('SntSubCl',
            '''
            <KON>                   # conjunction
            (
                [<AC><APPR>]?       # optional adverb or preposition ("außer wenn ...")
                <KOUS>              # subordinating conjunction
                %CONTENT%           # clause content
                (
                    [%VG%]          # because of possible tagging errors, we don't require verb
                                    # to be finite
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('SntSubCl',
            '''
            <SntSubCl>              # sentential subordinate clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    [%VG%]          # because of possible tagging errors, we don't require verb
                                    # to be finite
                )
                (?:
                    [<$,><KON>]
                    %CONTENT%       # clause content
                    [%VG%]          # because of possible tagging errors, we don't require verb
                                    # to be finite
                    [<$,>%EOS%]

                )*
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfSubCl',
            '''
            # (?:
            #     ^                   # start of sentence
            # |
            #     <$,>                # or comma
            # )
            [<AC><APPR>]?           # optional adverb or preposition ("außer um ...")
            <KOUI>                  # subordinating conjunction
            %CONTENT%               # clause content
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('InfSubCl',
            '''
            <KON>                   # conjunction
            (
                [<AC><APPR>]?       # optional adverb or preposition ("außer um ...")
                <KOUI>              # subordinating conjunction
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfSubCl',
            '''
            <InfSubCl>              # non-finite subordinate clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('IntCl',
            '''
            (?:
                <KON>              # or comma
            )?
            (
            [<PWS><PWAT><PWAV>]     # interrogative pronoun
            %CONTENT%               # clause content
            (
                [%VG%]              # verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%]?                  # optional end of sentence punctuation
            )
            ''',
            feats=lambda match: {'verb': match[1][0].get('verb')},
            level=12)

        add_rule('IntCl',
            '''
            <KON>
            (
                [<PWS><PWAT><PWAV>] # interrogative pronoun
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('IntCl',
            '''
            <IntCl>
            <KON>
            (
                %CONTENT%           # clause content
                (
                    [%VG%]          # verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
            group=1,
            feats=lambda match: {'verb': match[2][0].get('verb')},
            level=13)

        add_rule('InfCl',
            '''
            (
            (?:
                ^                   # start of a sentence
            |
                <$,>                # or comma
            |
                <KON>                # or conjunction
            )
            (?:
                [^<PRELS><PRELAT><KOUS><KOUI><FVG>]
                %CONTENT%
                [<NC><PC>]
                %CONTENT%
            )?
            (
                <IVG>               # non-finite verb group
            )
            %BASIC_TRAILER%?        # optional trailer
            [%EOS%%PUNCT%]?         # optional end of sentence punctuation
            )
            ''',
                 group = 1,
                 feats=lambda match: {'verb': match[2][0].get('verb')},
                 level=12)

        add_rule('InfCl',
            '''
            <InfCl>                 # non-finite (complement) clause
            <KON>                   # conjunction
            (
                %CONTENT%           # clause content
                (
                    <IVG>           # non-finite verb group
                )
                %BASIC_TRAILER%?    # optional trailer
                [%EOS%]?              # optional end of sentence punctuation
            )
            ''',
                 group = 1,
                 feats = lambda match: {'verb': match[2][0].get('verb')},
                 level = 13)

        ######################
        # Clause combination #
        ######################

        add_rule('InfCl',
            '''
            <InfCl>
            <RelCl>
            ''',
            level=14)

        add_rule('InfCl',
            '''
            <InfCl>
            (?:
                <KON>?
                <InfCl>
            )+
           ''',
            level=15)

        add_rule('IntCl',
            '''
            <IntCl>
            <RelCl>
            ''',
            level=14)

        add_rule('IntCl',
            '''
            <IntCl>
            (?:
                <KON>?
                <IntCl>
            )+
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <SntSubCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfSubCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            <InfCl>
            ''',
            level=15)

        add_rule('RelCl',
            '''
            <RelCl>
            (?:
                <KON>?
                <RelCl>
            )+
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <RelCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <IntCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            <InfSubCl>
            ''',
            level=16)

        add_rule('SntSubCl',
            '''
            <SntSubCl>
            (?:
                <KON>
                <SntSubCl>
            )+
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <RelCl>
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            <IntCl>
            ''',
            level=16)

        add_rule('InfSubCl',
            '''
            <InfSubCl>
            (?:
                <KON>
                <InfSubCl>
            )+
            ''',
            level=10)

        ################
        # Main clauses #
        ################

        # Verb-first or verb-second main clause

        def get_verb_feats(match):
            fin = match[1][0].get('verb')
            if fin is not None:
                if fin['pos'].startswith('VV'):
                    try:
                        particle = match[2][0]['lemma']
                    except IndexError:
                        particle = None
                    return {'verb': fin, 'verb_part': particle}
            try:
                inf = match[3][0].get('verb')
            except IndexError:
                return {'verb': fin}
            if inf is not None and inf['pos'].startswith(('VM', 'VV')):
                return {'verb': inf}
            return {'verb': fin}

        add_rule('MainCl',
            '''
                ^
                [^%VG%%CLAUSE%%DASH%%PUNCT%]*
                (?:
                [%VG%%CLAUSE%]
                [^%VG%%CLAUSE%%DASH%%PUNCT%]*
                )?
                (?: <$.> | <$,>)
            ''', level=17)

        add_rule('MainCl',
            '''
            <$,>
            <KON>
            (
                <SntSubCl>
            )
            [%CLAUSE%]*
            [%EOS%]?
            ''',
            feats=lambda match: match[1][0].feats,
            level=17)

        add_rule('MainCl',
            '''
            (
                <$,>
                <KON>
                (?:
                    [^<KON><MainCl><FVG>]+
                    (?:
                        <KON>?
                        [^<KON><MainCl><FVG>]+
                    )?
                )*
                (
                    <FVG>
                )
                [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                (?:
                    (?:
                        [%CLAUSE%]
                        (?:
                            <$,>
                            [%CLAUSE%]
                        )*
                    )?
                    [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                )*
                (
                    <PTKVZ>
                )?
                (
                    <IVG>
                )?
                %BASIC_TRAILER%?
                (?:
                    [%CLAUSE%]
                    (?:
                        <$,>
                        [%CLAUSE%]
                    )*
                )?
            )
            <$,>
            ''',
            group=1,
            level=17)

        add_rule('MainCl',
                 '''
                 (
                 (?:
                 ^
                 <KON>
                 )?
                 (?:
                 [^<KON><MainCl><IVG>]
                 (?:
                 <KON>
                 [^<KON><MainCl><IVG>]
                 )?
                 )*
                 (
                 <IVG>
                 )
                 [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                 (?:
                 [%CLAUSE%]
                 <$,>
                 [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                 )*
                 (
                 <PTKVZ>
                 )?
                 [%PUNCT%]
                 )
                 <AC>
                 ''', group=1, level=17)

        add_rule('MainCl',
                 '''
                 ^
                 (?:
                 <NE>+
                 (?: <$,> | <$(> )
                 )?
                 (
                 <SntSubCl>
                 )
                 [%CLAUSE%]+
                 <$.>?
                 ''',
                 feats=lambda match: match[1][0].feats,
                 level=17)

        add_rule('MainCl',
                 '''
                 (?:
                 ^
                 <KON>
                 )?
                 (?:
                 [^<KON><MainCl><FVG>%DASH%%PUNCT%]
                 (?:
                 <KON>
                 [^<KON><MainCl><FVG>]
                 )?
                 |
                 [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%]
                 )*
                 (
                 <FVG>
                 )
                 [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                 (?:
                 [%CLAUSE%]
                 [%PUNCT%]?
                 )*
                 [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%<PTKVZ>]*
                 (
                 <PTKVZ>
                 )?
                 (
                 # either a non-finite verb group
                 # or (error tolerance)
                 <IVG>
                 |
                 # a finite verb group if
                 <FVG>
                 # immediately followed by punctuation
                 [%PUNCT%]
                 )?
                 (?:
                 [%DASH%%PUNCT%]
                 (?:
                 [^%VG%%DASH%%PUNCT%]+
                 (?:
                 $
                 |
                 [%DASH%%PUNCT%]
                 )
                 )?
                 |
                 # commented because of errors
                 # [^%VG%<KON>%CLAUSE%%DASH%%PUNCT%]*
                 (?:
                 [%CLAUSE%]
                 <$,>?
                 )*
                 )?
                 (?: <$.> | <$,>)?
                 ''',
                 feats=get_verb_feats,
                 level=18)

        add_rule('MainCl',
                 '''
                 (
                 <MainCl>
                 )
                 (?:
                 [<KON><$,>]
                 <MainCl>
                 )+
                 ''',
                 feats=lambda match: match[1][0].feats,
                 level=19)

        add_rule('MainCl',
                 '''
                 <MainCl>
                 (
                 (
                 [^%VG%%CLAUSE%%DASH%%PUNCT%]
                 |
                 [^%VG%%CLAUSE%%DASH%%PUNCT%]
                 <$,>
                 )+
                 [%EOS%]
                 )
                 ''', group=1,
                 feats=lambda match: {'makeVerbLess': True},
                 level=20)

        # Catch-all rule (fallback).
        add_rule('ANY',
                 '''
                 [^<MainCl><SntSubCl>]+
                 ''',
                 level=21)
