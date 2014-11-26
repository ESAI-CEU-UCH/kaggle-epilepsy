#!/bin/bash

# This file is part of ESAI-CEU-UCH/kaggle-epilepsy (https://github.com/ESAI-CEU-UCH/kaggle-epilepsy)
#
# Copyright (c) 2014, ESAI, Universidad CEU Cardenal Herrera,
# (F. Zamora-Martínez, F. Muñoz-Malmaraz, P. Botella-Rocamora, J. Pardo)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#  
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

outdir=$LISTS_PATH

if [[ -z $outdir ]]; then
    echo -n "Unable to find environment variables, please check "
    echo -n "that scripts/conf.sh has been loaded by using: "
    echo -n ". scripts/env.sh"
    exit 10
fi
dirs=$1
if [ -z $dirs ]; then
    echo "Needs a list of directories"
    exit 10
fi
for dir in $dirs; do
    dir=$(readlink -f $dir)
    echo $dir
    for i in 1 2 3 4 5; do
        name=$outdir/Dog_${i}_$(basename $dir)
        train=$name.train
        test=$name.test
        if [ ! -e $train ]; then
            find $dir -name "Dog_${i}_*ictal*" | sed 's/.channel.*//g;s/.txt//g' |
            sort -u > $train
        fi
        if [ ! -e $test ]; then
            find $dir -name "Dog_${i}_*test*" | sed 's/.channel.*//g;s/.txt//g' |
            sort -u > $test
        fi
    done
    for i in 1 2; do
        name=$outdir/Patient_${i}_$(basename $dir)
        train=$name.train
        test=$name.test
        if [ ! -e $train ]; then
            find $dir -name "Patient_${i}_*ictal*" | sed 's/.channel.*//g;s/.txt//g' |
            sort -u > $train
        fi
        if [ ! -e $test ]; then
            find $dir -name "Patient_${i}_*test*" | sed 's/.channel.*//g;s/.txt//g' |
            sort -u > $test
        fi
    done
done
