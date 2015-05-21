#!/usr/bin/env python

##################################################################
# Libraries
from setuptools import setup
from os import path

##################################################################
# Variables and Constants
pwd = path.abspath(path.dirname(__file__))
with open(path.join(pwd, "README.md"), encoding="utf-8") as ifile:
    long_description = ifile.read()

##################################################################
# setup()
setup(name = "dsegmenter", version = "0.0.1dev1", \
          description = "discourse segmenters", \
          long_description = long_description, \
          author = "Wladimir Sidorenko (Uladzimir Sidarenka)", \
          author_email = "sidarenk@uni-potsdam.de", \
          license = "MIT", \
          url = "https://github.com/WladimirSidorenko/DiscourseSegmenter", \
          package_dir = {"": "lib"}, \
          packages = ["bpseg", "edseg", "treeseg"], \
          package_data = {}, \
          requires = ["scikit-learn (>=0.15.2)", \
                          "numpy (>=1.9.2)", \
                          "nltk (>=3.0.2)"], \
          provides = ["dsegmenter (0.0.1)"]
          scripts = [path.join("scripts", "discourse_segmenter")], \
          classifiers = ["Development Status :: 2 - Pre-Alpha", \
                             "Environment :: Console", \
                             "Intended Audience :: Science/Research", \
                             "License :: OSI Approved :: MIT License", \
                             "Natural Language :: German", \
                             "Operationg System :: Unix", \
                             "Operationg System :: MacOS", \
                             "Operationg System :: Microsoft :: Windows", \
                             "Programming Language :: Python :: 2", \
                             "Programming Language :: Python :: 2.6", \
                             "Programming Language :: Python :: 2.7", \
                             "Programming Language :: Python :: 3", \
                             "Topic :: Text Processing :: Linguistic"], \
          keywords="discourse segmentation NLP linguistics")
