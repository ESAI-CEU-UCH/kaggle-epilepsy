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

. settings.sh
mkdir -p $TMP_PATH
. scripts/configure.sh

cleanup()
{
    echo "CLEANING UP, PLEASE WAIT UNTIL FINISH"
    cd $ROOT_PATH
    for dest in "$@"; do
        rm -Rf $dest
    done
}

##################
## FFT FEATURES ##
##################

echo "Computing FFT features"
# Avoid FFT computation if not needed
if [[ ! -e $FFT_PATH ]]; then
    # Computes FFT features. Additionally it writes sequence numbers to
    # SEQUENCES_PATH.
    if ! $APRIL_EXEC scripts/PREPROCESS/compute_fft.lua $DATA_PATH $FFT_PATH $SEQUENCES_PATH; then
        echo "ERROR: Unable to compute FFT features"
        cleanup $FFT_PATH $SEQUENCES_PATH
        exit 10
    fi
fi

# Sanity check for the number of FFT files
N=$(find $FFT_PATH -name "*segment*csv.gz" | wc -l | awk '{print $1}')
if [[ $N -ne $NUMBER_FFT_FILES ]]; then
    echo "ERROR: Incorrect number of FFT files"
    cleanup $FFT_PATH $SEQUENCES_PATH
    exit 10
fi

###############################
## WINDOWED CORRELATIONS EIG ##
###############################

echo "Computing eigen values of windowed correlations matrix"
mkdir -p $WINDOWED_COR_PATH
if ! Rscript scripts/PREPROCESS/correlation_60s_30s.R; then
    echo "ERROR: Unable to compute eigen values of correlations matrix"
    cleanup $WINDOWED_COR_PATH
    exit 10
fi

###########################
## PCA OVER FFT FEATURES ##
###########################

echo "Computing PCA transformation of FFT features"
if [[ ! -e $PCA_TRANS_PATH ]]; then
    mkdir -p $PCA_TRANS_PATH
    if ! Rscript scripts/PREPROCESS/compute_pca.R; then
        echo "ERROR: Unable to compute PCA transformation"
        cleanup $PCA_TRANS_PATH
        exit 10
    fi
fi

echo "Applying PCA transformation to FFT features"
if [[ ! -e $FFT_PCA_PATH ]]; then
    mkdir -p $FFT_PCA_PATH
    if ! $APRIL_EXEC scripts/PREPROCESS/apply_pca.lua $FFT_PATH $PCA_TRANS_PATH $FFT_PCA_PATH; then
        echo "ERROR: Unable to apply PCA transformation to FFT data"
        cleanup $FFT_PCA_PATH
    fi
fi

###########################
## ICA OVER FFT FEATURES ##
###########################

echo "Computing and applying ICA transformation of FFT features"
if [[ ! -e $FFT_ICA_PATH ]]; then
    mkdir -p $FFT_ICA_PATH
    if ! Rscript scripts/PREPROCESS/compute_ica.R; then
        echo "ERROR: Unable to compute and apply ICA transformation"
        cleanup $FFT_ICA_PATH
        exit 10
    fi
fi
