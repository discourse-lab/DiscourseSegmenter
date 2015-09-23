# Discourse Segmenter

[![The MIT License](https://img.shields.io/dub/l/vibe-d.svg)](http://opensource.org/licenses/MIT)

A collection of various discourse segmenters with training data for German texts.

## Description

This python module currently comprises two discourse segmenters: *edseg* and *bparseg*.

The former segmenter is a rule-based system that uses shallow discourse-oriented parsing to determine boundaries of elementary discourse units in text.  The rules are hard-coded in the [submodule's file](dsegmenter/edseg/clause_segmentation.py) and are only applicable to German input.

The latter submodule is an ML-based segmenter that operates on syntactic constituency trees (output from [BitPar](http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/)) and decides whether a syntactic constituent initiates a discourse segment or not using a pre-trained linear SVM model.

## Installation

You can install this package:

* directly from source
```shell
git clone git@github.com:WladimirSidorenko/DiscourseSegmenter.git
cd DiscourseSegmenter
./setup.py install
```

* from the distributed tarball
```shell
pip install dsegmenter-0.0.1.dev1.tar.gz
```

## Usage

You can either import the python modules in your scripts (see an example [here](scripts/discourse_segmenter)), e.g.:

```python
from dsegmenter.bparseg import BparSegmenter

segmenter = BparSegmenter()
```

or use the stand-alone script that comes with this module:

```shell
discourse_segmenter --help
```
