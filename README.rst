===================
Discourse Segmenter
===================

.. image:: https://img.shields.io/badge/license-BSD-blue.svg
   :alt: BSD License
   :align: right
   :target: http://opensource.org/licenses/MIT

A collection of various discourse segmenters (primarily for German texts).

Description
===========

This python module currently comprises two discourse segmenters:
**edseg** and **bparseg**.

**edseg**
 is a rule-based system that uses shallow discourse-oriented
 parsing to determine boundaries of elementary discourse units in
 text.  The rules are hard-coded in the `submodule's file`_ and are
 only applicable to German input.

**bparseg**
 is an ML-based segmentation module that operates on
 syntactic constituency trees (output from BitPar_) and decides
 whether a syntactic constituent initiates a discourse segment or not
 using a pre-trained linear SVM model.  This model was trained on the
 German PCC_ corpus, but you can also train your own classifer for any
 language using your own training data (cf. ``discourse_segmenter
 --help`` for further instructions on how to do that).

*Since the current model is a serialized file and, therefore, likely  to be incompatible with future releases of `numpy`, we will probably  remove the model files from future versions of this package,  including source data instead and performing training during the  installation.*

Installation
============

To install this package from the distributed tarball (provided that
all requirements are satisfied), run

.. code-block:: shell

    pip install  https://github.com/WladimirSidorenko/DiscourseSegmenter/archive/0.0.1.dev1.tar.gz

Alternatively, you can also install it directly from the source
repository (currently recommended) by executing:

.. code-block:: shell

    git clone git@github.com:WladimirSidorenko/DiscourseSegmenter.git
    pip install -r DiscourseSegmenter/requirements.txt DiscourseSegmenter/ --user

Usage
=====

After installation, you can import the module in your python scripts
(see an example here_), e.g.:

.. code-block:: python

    from dsegmenter.bparseg import BparSegmenter

    segmenter = BparSegmenter()

or, alternatively, also use the delivered front-end script
`discourse_segmenter` to process your parsed input data, e.g.:

.. code-block:: shell

    discourse_segmenter bparseg segment DiscourseSegmenter/examples/bpar/maz-8727.exb.bpar

Note that this script requires two mandatory arguments: the type of
the segmenter to use (`bparseg` in the above case) and the operation
to perform (which are specific to each segmenter).

.. _`Bitpar`: http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/
.. _`PCC`: http://www.lrec-conf.org/proceedings/lrec2014/pdf/579_Paper.pdf
.. _`here`: https://github.com/WladimirSidorenko/DiscourseSegmenter/blob/master/scripts/discourse_segmenter
.. _`submodule's file`: https://github.com/WladimirSidorenko/DiscourseSegmenter/blob/master/dsegmenter/edseg/clause_segmentation.py
