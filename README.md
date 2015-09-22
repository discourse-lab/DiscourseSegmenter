# Discourse Segmenter

A collection of various discourse segmenters with training data for German texts.

## Description

This python module currently comprises two discourse segmenters: *edseg* and *bparseg*.

The former submodule is a rule-based system that uses shallow discourse-oriented parsing to determine boundaries of elementary discourse units in text.  The rules are hard-coded in the [submodule's file](dsegmenter/edseg/clause_segmentation.py) and are only applicable to German input.

The *bparseg* is an ML-based segmenter that operates on syntactic constituency trees (output from [BitPar](http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/)) and decides whether a syntactic constituent initiates a discourse segment or not using a pre-trained linear SVM model.

## Installation

### From source tarball

```
pip install dsegmenter-0.0.1dev1.tgz
```

## Usage

You can either import the python modules in your scripts (see an example [here](scripts/discourse_segmenter)), e.g.:

```
from dsegmenter.bparseg import BparSegmenter, CTree

segmenter = BparSegmenter()
```

or use the stand-alone script that comes with this module:

```
discourse_segmenter --help
```
