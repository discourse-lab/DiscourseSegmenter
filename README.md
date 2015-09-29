# Discourse Segmenter

[![The MIT License](https://img.shields.io/dub/l/vibe-d.svg)](http://opensource.org/licenses/MIT)

A collection of various discourse segmenters (primarily for German texts).

## Description

This python module currently comprises two discourse segmenters: *edseg* and *bparseg*.

*edseg* is a rule-based system that uses shallow discourse-oriented parsing to determine boundaries of elementary discourse units in text.  The rules are hard-coded in the [submodule's file](dsegmenter/edseg/clause_segmentation.py) and are only applicable to German input.

*bparseg* is an ML-based segmentation module that operates on syntactic constituency trees (output from [BitPar](http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/)) and decides whether a syntactic constituent initiates a discourse segment or not using a pre-trained linear SVM model.  This model was trained on the German [PCC](http://www.lrec-conf.org/proceedings/lrec2014/pdf/579_Paper.pdf) corpus, but you can also train your own classifer for any language using your own training data (cf. `discourse_segmenter --help` for further instructions on how to do that).

**Since the current model is a serialized file and, therefore, likely to be incompatible with future releases of `numpy`, we will probably remove it from future releases, including source data instead and performing training during the installation.**

## Installation

To install this package from the distributed tarball, run
```shell
pip install  https://github.com/WladimirSidorenko/DiscourseSegmenter/archive/0.0.1.dev1.tar.gz
```

Alternatively, you can also install it directly from the source repository by executing:
```shell
git clone git@github.com:WladimirSidorenko/DiscourseSegmenter.git
cd DiscourseSegmenter
./setup.py install
```

## Usage

After installation, you can import the module in your python scripts (see an example [here](scripts/discourse_segmenter)), e.g.:

```python
from dsegmenter.bparseg import BparSegmenter

segmenter = BparSegmenter()
```

or, alternatively, also use the delivered front-end script `discourse_segmenter` to process your parsed input data, e.g.:

```shell
discourse_segmenter examples/bpar/maz-8727.exb.bpar
```
