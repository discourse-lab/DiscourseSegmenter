#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Libraries
from setuptools import setup
from os import path
import codecs


##################################################################
# Variables and Constants
PWD = path.abspath(path.dirname(__file__))
ENCODING = "utf-8"

with codecs.open(path.join(PWD, "README.rst"), encoding="utf-8") as ifile:
    long_description = ifile.read()

INSTALL_REQUIRES = []
with codecs.open(path.join(PWD, "requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            INSTALL_REQUIRES.append(iline)

TEST_REQUIRES = []
with codecs.open(path.join(PWD, "test-requirements.txt"),
                 encoding=ENCODING) as ifile:
    for iline in ifile:
        iline = iline.strip()
        if iline:
            TEST_REQUIRES.append(iline)

##################################################################
# setup()
setup(
    name="dsegmenter",
    version="0.2.0",
    description=("Collection of discourse segmenters "
                 "(with pre-trained models for German)"),
    long_description=long_description,
    author="Wladimir Sidorenko (Uladzimir Sidarenka)",
    author_email="sidarenk@uni-potsdam.de",
    license="MIT",
    url="https://github.com/discourse-lab/DiscourseSegmenter",
    include_package_data=True,
    packages=["dsegmenter", "dsegmenter.bparseg", "dsegmenter.edseg",
              "dsegmenter.treeseg", "dsegmenter.mateseg",
              "dsegmenter.evaluation"],
    package_data={
        "dsegmenter.edseg": [path.join("data", fname) for fname in (
            "dass_verbs.txt", "discourse_preps.txt", "finite_verbs.txt",
            "reporting_verbs.txt", "skip_rules.txt")],
        "dsegmenter.bparseg": [path.join("data", "*.npy"),
                               path.join("data", "*.model")],
        "dsegmenter.mateseg": [path.join("data", "mate.model")]},
    install_requires=INSTALL_REQUIRES,
    setup_requires=["pytest-runner"],
    tests_require=TEST_REQUIRES,
    provides=["dsegmenter (0.2.0)"],
    scripts=[path.join("scripts", "discourse_segmenter"),
             path.join("scripts", "evaluation")],
    classifiers=["Development Status :: 4 - Beta",
                 "Environment :: Console",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: MIT License",
                 "Natural Language :: German",
                 "Operating System :: Unix",
                 "Operating System :: MacOS",
                 # "Operating System :: Microsoft :: Windows",
                 "Programming Language :: Python :: 2",
                 "Programming Language :: Python :: 2.6",
                 "Programming Language :: Python :: 2.7",
                 # "Programming Language :: Python :: 3",
                 "Topic :: Text Processing :: Linguistic"],
    keywords="discourse segmentation NLP linguistics")
