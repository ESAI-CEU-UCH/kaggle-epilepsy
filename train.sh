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
ANN2_ICA_CORW_CONF=scripts/MODELS/confs/ann2_ica_corw.lua
ANN2P_PCA_CORW_CONF=scripts/MODELS/confs/ann2p_pca_corw.lua
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

train()
{
    SCRIPT=$1
    CONF=$2
    RESULT=$3
    ARGS=$4
    # trap keyboard interrupt (control-c)
    trap "control_c $RESULT" SIGINT
    #
    mkdir -p $RESULT
    echo clip,preictal > $RESULT/test.txt
    echo "Training with script= $SCRIPT   conf= $CONF   result= $RESULT   args= $ARGS"
    echo "IT CAN TAKE SEVERAL HOURS, PLEASE WAIT"
    if [[ $VERBOSE_TRAIN == 0 ]]; then
	$APRIL_EXEC $TRAIN_ALL_SCRIPT $SCRIPT -f $CONF $ARGS \
            --test=$RESULT/test.txt \
            --prefix=$RESULT > $RESULT/train.out
	err=$?
    else
	$APRIL_EXEC $TRAIN_ALL_SCRIPT $MLP_TRAIN_SCRIPT -f $CONF $ARGS \
	    
            --fft=$FFT_PCA_PATH --cor=$WINDOWED_COR_PATH \
            --test=$RESULT/test.txt \
            --prefix=$RESULT | tee $RESULT/train.out
	err=$?
    fi
    # removes keyboard interrupt trap (control-c)
    trap - SIGINT
    return $err    
}

train_mlp_pca()
{
    train $MLP_TRAIN_SCRIPT $1 $2 "--fft=$FFT_PCA_PATH --cor=$WINDOWED_COR_PATH"
    return $?
}

train_mlp_ica()
{
    train $MLP_TRAIN_SCRIPT $1 $2 "--fft=$FFT_ICA_PATH --cor=$WINDOWED_COR_PATH"
    return $?
}

###############################################################################

if [[ ( ! -e $FFT_PCA_PATH ) || ( ! -e $FFT_ICA_PATH ) || ( ! -e $WINDOWED_COR_PATH ) ]]; then
    if ! ./preprocess.sh; then
        exit 10
    fi
fi

###################
## ANN5 PCA+CORW ##
###################

if [[ ! -e $ANN5_PCA_CORW_RESULT ]]; then
    if ! train_mlp_pca $ANN5_PCA_CORW_CONF $ANN5_PCA_CORW_RESULT; then
        cleanup $ANN5_PCA_CORW_RESULT
	exit 10
    fi
fi

####################
## ANN2p PCA+CORW ##
####################

if [[ ! -e $ANN2P_PCA_CORW_RESULT ]]; then
    if ! train_mlp_pca $ANN2P_PCA_CORW_CONF $ANN2P_PCA_CORW_RESULT; then
        cleanup $ANN2P_PCA_CORW_RESULT
	exit 10
    fi
fi

###################
## ANN2 ICA+CORW ##
###################

if [[ ! -e $ANN2_ICA_CORW_RESULT ]]; then
    if ! train_mlp_ica $ANN2_ICA_CORW_CONF $ANN2_ICA_CORW_RESULT; then
        cleanup $ANN2_ICA_CORW_RESULT
	exit 10
    fi
fi
