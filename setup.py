#!/usr/bin/env python

##################################################################
# Libraries
from distutils.core import setup
from os import path
import codecs
import glob

##################################################################
# Variables and Constants
pwd = path.abspath(path.dirname(__file__))
with codecs.open(path.join(pwd, "README.rst"), encoding="utf-8") as ifile:
    long_description = ifile.read()

##################################################################
# setup()
setup(name = "dsegmenter", version = "0.0.1.dev1", \
          description = "discourse segmenters", \
          long_description = long_description, \
          author = "Wladimir Sidorenko (Uladzimir Sidarenka)", \
          author_email = "sidarenk@uni-potsdam.de", \
          license = "MIT", \
          url = "https://github.com/WladimirSidorenko/DiscourseSegmenter", \
          include_package_data = True,
          packages = ["dsegmenter", "dsegmenter.bparseg", "dsegmenter.edseg", \
                          "dsegmenter.treeseg"], \
          # package_dir = {"dsegmenter.bparseg": "dsegmenter", \
          #                    "dsegmenter.edseg": "dsegmenter", \
          #                    "dsegmenter.treeseg": "dsegmenter"}, \
          package_data = {"dsegmenter.edseg": [path.join("data", fname) for fname in \
                                        ("dass_verbs.txt", "discourse_preps.txt", \
                                             "finite_verbs.txt", "reporting_verbs.txt", \
                                             "skip_rules.txt")], \
                          "dsegmenter.bparseg": [path.join("data", "*.npy"), \
                                                          path.join("data", "*.model")]}, \
          requires = ["scikit.learn (>=0.15.2)", \
                          "numpy (>=1.9.2)", \
                          "nltk (>=3.0.2)"], \
          provides = ["dsegmenter (0.0.1)"], \
          scripts = [path.join("scripts", "discourse_segmenter")], \
          classifiers = ["Development Status :: 2 - Pre-Alpha", \
                             "Environment :: Console", \
                             "Intended Audience :: Science/Research", \
                             "License :: OSI Approved :: MIT License", \
                             "Natural Language :: German", \
                             "Operating System :: Unix", \
                             "Operating System :: MacOS", \
                             "Operating System :: Microsoft :: Windows", \
                             "Programming Language :: Python :: 2", \
                             "Programming Language :: Python :: 2.6", \
                             "Programming Language :: Python :: 2.7", \
                             "Programming Language :: Python :: 3", \
                             "Topic :: Text Processing :: Linguistic"], \
          keywords="discourse segmentation NLP linguistics")
