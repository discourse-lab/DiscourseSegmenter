#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation
"""This module provides a convenient interface for handling CONLL data.

CONLL data are represented in the form of individual lines with tab-separated
fields.  This module provides several classes which parse such lines either
incrementally, one by one, or all at once, and store their information in their
internal data structure.

Attributes:
  EOS (str): end of sentence marker
  EOL (str): end of line marker
  EOS_TAG (str): end tag for sentences
  FIELDSEP (str): separator of fields for description of a single word
  EMPTY_FIELD (str): word denoting an empty fields in the word
  FEAT_SEP (str): separator of individual features
  FEAT_VALUE_SEP (str): separator of feature name and its value
  FEAT_VALUE_SEP_RE (re): regular expression corresponding to FEAT_VALUE_SEP

Classes:
  CONLL: class for handling CONLL forrests
  CONLLSentence: class storing information pertaining to a single CONLL
    sentence
  CONLLWord: class storing information about a single CONLL word

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Loaded Modules
from collections import defaultdict
import re

##################################################################
# Interface
__name__ = "conll"
__all__ = ["EOS", "EOL", "EOS_TAG", "FIELDSEP", "EMPTY_FIELD",
           "FEAT_SEP", "FEAT_VALUE_SEP", "FEAT_VALUE_SEP_RE",
           "CONLL", "CONLLSentence", "CONLLWord"]

##################################################################
# Constants
EOS = '\n'
EOL = '\n'
EOS_TAG = "<s />"
FIELDSEP = '\t'
EMPTY_FIELD = '_'

FEAT_SEP = '|'
FEAT_VALUE_SEP = '='
FEAT_VALUE_SEP_RE = re.compile(FEAT_VALUE_SEP)


##################################################################
# Classes
class CONLL(object):
    """
    Class for storing and manipulating CONLL parse forrest information.

    An instance of this class comprises information about one or multiple
    parsed sentences in CONLL format.

    """

    def __init__(self, istring=''):
        """Initialize instance variables and parse input string if specified.

        Args:
          istring (basestring): input string(s) with CONLL data (optional)

        """
        self.s_id = -1
        self.sentences = []
        self.__eos_seen__ = True
        for iline in istring.splitlines():
            self.add_line(iline)

    def add_line(self, iline=u''):
        """Parse line and add it as CONLL word.

        Args:
          iline (basestring): input line(s) to parse

        Returns:
          void:

        """
        iline = iline.strip()
        if not iline or iline == EOS or iline == EOS_TAG:
            # if we see a line which appears to be the end of a sentence, we
            # simply set corresponding flag
            self.__eos_seen__ = True
        elif self.__eos_seen__:
            # otherwise, if end of sentence has been seen before and the line
            # appears to be non-empty, increase the counter of sentences and
            # append next sentence to the list
            self._add_sentence(CONLLWord(iline))
            self.__eos_seen__ = False
        else:
            # otherwise, parse line as a CONLL word and compare its index to
            # the index of last parsed CONLL sentence. If the index of the new
            # word is less than the index of the last word, that means that a
            # new sentence has started.
            w = CONLLWord(iline)
            if self.s_id == -1 or \
               int(w.idx) < int(self.sentences[self.s_id].words[-1].idx):
                self._add_sentence(w)
            else:
                self.sentences[self.s_id].push_word(w)

    def is_empty(self):
        """Check whether any sentences are stored.

        Returns:
          bool: True if there is at least one sentence.

        """
        return self.s_id == -1

    def clear(self):
        """
        Remove all stored information.
        """
        del self.sentences[:]
        self.s_id = -1
        self.__eos_seen__ = False

    def get_words(self):
        """
        Return list of all words wird indices from all sentences.

        Return a list of all words from all sentences in consecutive order as
        tuples with three elements (word, sentence_idx, word_idx) where the
        first element is a word, the next element is its index in the list of
        sentences, and the third element is word's index within the sentence.

        """
        retlist = []
        for s_id in xrange(self.s_id + 1):
            retlist += [(w.form, s_id, w_id) for w, w_id in
                        self.sentences[s_id].get_words()]
        return retlist

    def __unicode__(self):
        """Return unicode representation of current object."""
        ostring = u'\n'.join([unicode(s) for s in self.sentences])
        return ostring

    def __str__(self):
        """Return string representation of this object encoded in UTF-8."""
        return self.__unicode__().encode("utf-8")

    def __getitem__(self, i):
        """Return reference to `i`-th sentence in forrest.

        Args:
          i (int): integer index of sentence in forrest

        Returns:
          CONLLSentence: `i`-th CONLL sentence in forrest.

        Raises:
          IndexError: is raised if `i` is outside of forrest boundaries.

        """
        return self.sentences[i]

    def __setitem__(self, i, value):
        """Set `i`-th sentence in forrest to specified value.

        Args:
          i (int): integer index of sentence in forrest
          value (CONLLSentence): to which i-th sentence should be set

        Returns:
          CONLLSentence:new value of `i`-th sentence

        Raises:
          IndexError: raised if `i` is outside of forrest boundaries.

        """
        self.sentences[i] = value
        return self.sentences[i]

    def __iter__(self):
        """Return iterator object over sentences."""
        for s in self.sentences:
            yield s

    def _add_sentence(self, iword):
        """Add new sentence populating it with iword."""
        self.s_id += 1
        self.sentences.append(CONLLSentence(iword))


class CONLLSentence(object):
    """
    Class for storing and manipulating a single CONLL sentence.

    An instance of this class comprises information about a single sentence in
    CONLL format.

    This class provides following instance variables:
    self.words - list of all words belonging to given sentence
    self.w_id  - index of last word in self.words
    self.children  - index of last word in self.words

    This class provides following public methods:
    __init__()   - class constructor
    self.clear() - remove all words and reset counters
    self.is_empty() - check if any words are present in sentence
    self.push_word() - add given CONLLWord to sentence's list of words
    self.get_words() - return list of words with their indices
    __str__() - return string representation of sentence
    __unicode__() - return UNICODE representation of sentence
    __iter__() - return an iterator object over words
    __getitem__() - return word from sentence
    __setitem__() - set word in sentence
    __reversed__() - retun a reverse iterator over words
    __len__() - return the number of words in sentence

    """

    def __init__(self, iword=""):
        """Initialize instance variables and parse iline if specified."""
        self.w_id = -1
        self.words = []
        self.children = defaultdict(list)
        if iword:
            self.push_word(iword)

    def clear(self):
        """Remove all words and reset counters."""
        self.w_id = -1
        self.children.clear()
        del self.words[:]

    def is_empty(self):
        """Check if any words are present in sentence."""
        return self.w_id == -1

    def push_word(self, iword):
        """Parse iline storing its information in instance variables."""
        self.w_id += 1
        self.words.append(iword)
        self.children[iword.phead].append(self.words[self.w_id])

    def get_words(self):
        """
        Return list of all words with their indices.

        Return a list of all words in this sentence in consecutive order as
        tuples with two elements where the first element is the word itself and
        second element is its index within the sentence.
        """
        return zip(self.words, xrange(self.w_id + 1))

    def __unicode__(self):
        """Return string representation of this object."""
        ostring = EOL.join([unicode(w) for w in self.words]) + EOS
        return ostring

    def __str__(self):
        """Return string representation of this object encoded in UTF-8."""
        return self.__unicode__().encode("utf-8")

    def __iter__(self):
        """Return iterator object over words."""
        for w in self.words:
            yield w

    def __reversed__(self):
        """Return iterator object over words."""
        for w in self.words[::-1]:
            yield w

    def __getitem__(self, i):
        """
        Return reference to `i`-th word in sentence.

        @param i - integer index of word in sentence

        @return value of `i`-th word in sentence. IndexError is raised if `i`
        is outside of sentence boundaries.

        """
        return self.words[i]

    def __setitem__(self, i, value):
        """Set `i`-th word in sentence to specified value.

        @param i - integer index of sentence in forrest
        @param value - CONLL word to which i-th instance should be set

        @return new value of `i`-th word. IndexError is raised if `i` is
        outside of sentence boundaries.

        """
        self.words[i] = value
        return self.words[i]

    def __len__(self):
        """Return the number of words in sentence."""
        return len(self.words)


class CONLLWord(object):

    """Class for storing and manipulating information about a single word.

    An instance of this class comprises information about one word of CONLL
    tree.

    This class provides following static variables:
    key2field - mapping from attribute name to its position in attribute list
    REQFIELDS   - number of fields which has to be specified for a word

    This class provides following instance variables:
    self.fields - list of all word's attributes as they are defined in fields
    self.features - dictionary of features

    This class provides following public methods:
    __init__()      - class constructor
    self.parse_line() - parse specified CONLL line and populate instance
                      variables correspondingly
    add_features()  - update dictionary of features from another dictionary
    get()           - safe method for accessing missing attributes
    __getattr__()   - this method returns `self.field`s item if the name of
                      attribute is found in `key2field`
    __getitem__()  - this method allows access to CONLLWord field using
                     the standard dictionary like syntax, e.g. iword["token]
    __setitem__()   - this method allows to set values of CONLLWord fields by
                      using the dictionary like syntax,
                      e.g., iword["token] = "sky"
    __str__()       - return string representation of current forrest

    """

    key2field = {'idx': 0, 'form': 1, 'pform': 2, 'lemma': 3, 'plemma': 4,
                 'pos': 5, 'ppos': 6, 'feat': 7, 'head': 8, 'phead': 9,
                 'deprel': 10, 'pdeprel': 11, 'fillpred': 12, 'pred': 13}
    REQFIELDS = len(key2field)

    def __init__(self, iline=None):
        """Initialize instance variables and parse iline if specified."""
        self.fields = []
        self.features = {}
        if iline:
            self.parse_line(iline)

    def parse_line(self, iline):
        """Parse iline storing its information in instance variables."""
        self.fields = iline.split(FIELDSEP)
        nfields = len(self.fields)
        # check that proper number of fields is provided
        if nfields != self.REQFIELDS:
            raise Exception(
                "Incorrect line format ({:d} fields"
                " expected instead of {:d}):\n'{:s}'".format(
                    self.REQFIELDS, nfields, iline))
        # convert features and pfeatures to dicts
        feat_i = CONLLWord.key2field["feat"]
        self.features = self.fields[feat_i] = \
            self._str2dict(self.fields[feat_i])

    def add_features(self, newfeatures={}):
        """Update dictionary of features with new features."""
        self.features.update(newfeatures)

    def get(self, ikey, idefault=None):
        """Return value of ikey field or idefault if the field is missing."""
        try:
            return self.__getattr__(ikey)
        except AttributeError:
            return idefault

    def __contains__(self, name):
        """Check if field is present in item.

        This method looks for the passed field name in `key2field` dict and
        returns true if the name is found and false otherwise.

        Args:
        name (str): name of the field to be retrieved

        Returns:
        (bool):
        true if the given field name is found in item

        """
        return name in self.key2field

    def __getattr__(self, name):
        """Return field's item if this item's name is present in key2field.

        This method looks for passed name in `key2field` dict and returns
        corresponding item of `self.fields` or raises an AttributeException
        if no such item was found.

        @param name - name of the field to be retrieved

        """
        if name in self.key2field:
            return self.fields[self.key2field[name]]
        else:
            raise AttributeError("Cannot find symbol {:s}".format(name))

    def __getitem__(self, name):
        """Return field's item if this item's name is present in key2field.

        This method uses the self.__getattr__() method but converts the
        AttributeException to IndexError in case when lookup was not
        successful.

        @param name - name of the field to be retrieved

        """
        try:
            return self.__getattr__(name)
        except AttributeError:
            raise IndexError("cannot find index {:s}".format(name))

    def __setitem__(self, name, value):
        """Set the value of given item `name' to `value'.

        @param name - name of the attribute to be set
        @param value - new value of the attribute

        """
        if name in self.key2field:
            self.fields[self.key2field[name]] = value
        else:
            raise IndexError("cannot find index {:s}".format(name))

    def __unicode__(self):
        """Return unicode representation of this object."""
        retStr = u''
        # convert features and pfeatures to strings
        feat_i = CONLLWord.key2field["feat"]
        feat_str = self._dict2str(self.fields[feat_i])
        # construct return string (we can't change feature dictionary in place
        # (because next call to __str__() would be invalid), so slicing is
        # needed)
        retStr += FIELDSEP.join(self.fields[:feat_i])
        # add feature string
        if feat_i > 0:
            retStr += FIELDSEP
        retStr += feat_str
        if feat_i < self.REQFIELDS:
            retStr += FIELDSEP
        # add the rest of the fields
        retStr += FIELDSEP.join(self.fields[feat_i + 1:])
        return retStr

    def __str__(self):
        """Return string representation of this object encoded in UTF-8."""
        return self.__unicode__().encode("utf-8")

    def _dict2str(self, idict, new_format=True):
        """Convert dictionary of features to a string."""
        fList = []
        if not idict:
            return EMPTY_FIELD
        for fname, fvalue in idict.iteritems():
            if new_format:
                fList.append(fvalue)
            else:
                fList.append(fname + FEAT_VALUE_SEP + fvalue)
        return FEAT_SEP.join(fList)

    def _str2dict(self, istring):
        """Convert string of features to a dictionary."""
        retDict = {}
        if istring == EMPTY_FIELD:
            return retDict
        for feat in istring.split(FEAT_SEP):
            # feature format changed in MATE
            if FEAT_VALUE_SEP_RE.search(feat):
                retDict.update((feat.split(FEAT_VALUE_SEP),))
            else:
                retDict.update([self._new2old(feat)])
        return retDict

    def _new2old(self, ifeat):
        """Translate new representation of features to the old one
        @param ifeat - feature value

        @return  2-tuple of key value pair
        """
        ifeat = ifeat.lower()
        if ifeat in set(["nom", "gen", "dat", "acc"]):
            return ("case", ifeat)
        elif ifeat in set(["fem", "masc", "neut"]):
            return ("gender", ifeat)
        elif ifeat in set(["sg", "pl"]):
            return ("num", ifeat)
        elif ifeat in set(["1", "2", "3"]):
            return ("pers", ifeat)
        elif ifeat in set(["ind", "imp", "subj"]):
            return ("mood", ifeat)
        elif ifeat in set(["pres", "past"]):
            return ("tense", ifeat)
        return (ifeat, "True")
