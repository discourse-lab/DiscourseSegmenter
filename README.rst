===================
Discourse Segmenter
===================

.. image:: https://travis-ci.org/WladimirSidorenko/DiscourseSegmenter.svg?branch=master
   :alt: Build Status
   :align: right
   :target: https://travis-ci.org/WladimirSidorenko/DiscourseSegmenter

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :alt: MIT License
   :align: right
   :target: http://opensource.org/licenses/MIT

A collection of various discourse segmenters (with pre-trained models for German texts).


Description
===========

This python module currently comprises three discourse segmenters:
**edseg**, **bparseg**, and **mateseg**.

**edseg**
 is a rule-based system that uses shallow discourse-oriented
 parsing to determine the boundaries of elementary discourse units.
 The rules are hard-coded in the `submodule's file`_ and are
 only applicable to German input.

**bparseg**
 is an ML-based segmentation module that operates on
 syntactic constituency trees (output from BitPar_) and decides
 whether a syntactic constituent initiates a discourse segment or not
 using a pre-trained linear SVM model.  This model was trained on the
 German PCC_ corpus, but you can also train your own classifer for any
 language using your own training data (cf. ``discourse_segmenter
 --help`` for further instructions on how to do that).

**mateseg**
 is another ML-based segmentation module that operates on dependency
 trees (output from MateParser_) and decides whether a sub-structure
 of the dependency graph initiates a discourse segment or not using
 a pre-trained linear SVM model.  Again, this model was trained on
 the German PCC_ corpus.


Installation
============

To install this package from the PyPi index, run

.. code-block:: shell

    pip install dsegmenter

Alternatively, you can also install it directly from the source
repository by executing:

.. code-block:: shell

    git clone git@github.com:discourse-lab/DiscourseSegmenter.git
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

or

.. code-block:: shell

    discourse_segmenter mateseg segment DiscourseSegmenter/examples/conll/maz-8727.parsed.conll

Note that this script requires two mandatory arguments: the type of
the segmenter to use (`bparseg` or `mateseg` in the above cases) and the
operation to perform (which meight be specific to each segmenter).


Evaluation
==========

Intrinsic evaluation scores of the machine learning models on the
predicted vectors will be printed when training and evaluating a
segmentation model.

Extrinsic evaluation scores on the predicted segmentation trees can be
calculated with the evaluation script.

.. code-block:: shell

    evaluation {FOLDER:TRUE} {FOLDER:PRED}

Note, that the script internally calls the `DKpro agreement library`_,
which requires Java 8.



.. _`Bitpar`: http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/
.. _`MateParser`: http://code.google.com/p/mate-tools/
.. _`PCC`: http://www.lrec-conf.org/proceedings/lrec2014/pdf/579_Paper.pdf
.. _`here`: https://github.com/discourse-lab/DiscourseSegmenter/blob/master/scripts/discourse_segmenter
.. _`submodule's file`: https://github.com/discourse-lab/DiscourseSegmenter/blob/master/dsegmenter/edseg/clause_segmentation.py
.. _`DKpro agreement library`: https://dkpro.github.io/dkpro-statistics/
