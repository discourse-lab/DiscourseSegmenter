#!/bin/sh

set -e

arg="$(readlink -f $1)"
rundir="$(dirname $0)"
cd "${rundir}"
java -classpath .:dkpro-statistics-agreement-2.1.0.jar UnitizingAnnotationEvaluator "${arg}"
