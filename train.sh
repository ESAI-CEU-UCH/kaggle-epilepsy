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
. scripts/configure.sh

ANN5_PCA_CORW_CONF=scripts/MODELS/confs/ann5_pca_corw.lua
TRAIN_ALL_SCRIPT=scripts/MODELS/train_all_subjects_wrapper.lua
MLP_TRAIN_SCRIPT=scripts/MODELS/train_one_subject_mlp.lua
KNN_TRAIN_SCRIPT=scripts/MODELS/train_one_subject_knn.lua

###############################################################################

cleanup()
{
    echo "CLEANING UP, PLEASE WAIT UNTIL FINISH"
    cd $ROOT_PATH
    for dest in "$@"; do
        rm -Rf $dest
    done
}

# control-c execution
control_c()
{
    echo -en "\n*** Exiting by control-c ***\n"
    cleanup $1
    exit 10
}

train_mlp()
{
    CONF=$1
    RESULT=$2
    # trap keyboard interrupt (control-c)
    trap 'control_c $RESULT' SIGINT
    #
    mkdir -p $RESULT
    $APRIL_EXEC $TRAIN_ALL_SCRIPT $MLP_TRAIN_SCRIPT -f $CONF \
        --fft=$FFT_PCA_PATH --cor=$WINDOWED_COR_PATH \
        --test=$RESULT/test.txt \
        --prefix=$RESULT > $RESULT/train.out 2> $RESULT/train.err    
    err=$?
    # removes keyboard interrupt trap (control-c)
    trap - SIGINT
    return $err
}

###############################################################################

if [[ \( ! -e $FFT_PCA_PATH \) || \( ! -e $FFT_ICA_PATH \) || \( ! -e $WINDOWED_COR_PATH \) ]]; then
    if ! ./preprocess.sh; then
        exit 10
    fi
fi

###################
## ANN5 PCA+CORW ##
###################

if [[ ! -e $ANN5_PCA_CORW_RESULT ]]; then
    if ! train_mlp $ANN5_PCA_CORW_CONF $ANN5_PCA_CORW_RESULT; then
        cleanup $ANN5_PCA_CORW_RESULT
    fi
fi

####################
## ANN2p PCA+CORW ##
####################

if [[ ! -e $ANN2P_PCA_CORW_RESULT ]]; then
    if ! train_mlp $ANN2P_PCA_CORW_CONF $ANN2P_PCA_CORW_RESULT; then
        cleanup $ANN2P_PCA_CORW_RESULT
    fi
fi
