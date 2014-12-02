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
KNN_ICA_CONF=scripts/MODELS/confs/knn_ica.lua
KNN_PCA_CONF=scripts/MODELS/confs/knn_pca.lua
KNN_CORG_CONF=scripts/MODELS/confs/knn_corg.lua
KNN_COVRED_CONF=scripts/MODELS/confs/knn_covred.lua
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
    echo "Training with script= $SCRIPT   conf= $CONF   result= $RESULT   args= \"$ARGS\""
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

train_knn_pca()
{
    train $KNN_TRAIN_SCRIPT $1 $2 "--fft=$FFT_PCA_PATH --cor=$WINDOWED_COR_PATH"
    return $?
}

train_knn_ica()
{
    train $KNN_TRAIN_SCRIPT $1 $2 "--fft=$FFT_ICA_PATH --cor=$WINDOWED_COR_PATH"
    return $?
}

train_knn_corg()
{
    train $KNN_TRAIN_SCRIPT $1 $2 "--fft=$CORG_PATH"
    return $?
}

train_knn_covred()
{
    train $KNN_TRAIN_SCRIPT $1 $2 "--fft=$COVRED_PATH"
    return $?
}

bmc_ensemble()
{
    $APRIL_EXEC scripts/ENSEMBLE/bmc_ensemble.lua test.txt \
        $ANN2P_PCA_CORW_RESULT $ANN5_PCA_CORW_RESULT $ANN2_ICA_CORW_RESULT \
        $KNN_ICA_CORW_RESULT $KNN_PCA_CORW_RESULT \
        $KNN_CORG_RESULT $KNN_COVRED_RESULT > $BMC_ENSEMBLE_RESULT/test.txt 2> $BMC_ENSEMBLE_RESULT/cv.log
    return $?
}

###############################################################################

if [[ ( ! -e $FFT_PCA_PATH ) || ( ! -e $FFT_ICA_PATH ) || ( ! -e $WINDOWED_COR_PATH ) || ( ! -e $CORG_PATH ) || ( ! -e $COVRED_PATH ) ]]; then
    if ! ./preprocess.sh; then
        exit 10
    fi
fi

###################
## ANN5 PCA+CORW ##
###################

if [[ ! -e $ANN5_PCA_CORW_RESULT ]]; then
    mkdir -p $ANN5_PCA_CORW_RESULT
    if ! train_mlp_pca $ANN5_PCA_CORW_CONF $ANN5_PCA_CORW_RESULT; then
        cleanup $ANN5_PCA_CORW_RESULT
	exit 10
    fi
fi

####################
## ANN2p PCA+CORW ##
####################

if [[ ! -e $ANN2P_PCA_CORW_RESULT ]]; then
    mkdir -p $ANN2P_PCA_CORW_RESULT
    if ! train_mlp_pca $ANN2P_PCA_CORW_CONF $ANN2P_PCA_CORW_RESULT; then
        cleanup $ANN2P_PCA_CORW_RESULT
	exit 10
    fi
fi

###################
## ANN2 ICA+CORW ##
###################

if [[ ! -e $ANN2_ICA_CORW_RESULT ]]; then
    mkdir -p $ANN2_ICA_CORW_RESULT
    if ! train_mlp_ica $ANN2_ICA_CORW_CONF $ANN2_ICA_CORW_RESULT; then
        cleanup $ANN2_ICA_CORW_RESULT
	exit 10
    fi
fi

##################
## KNN PCA+CORW ##
##################

if [[ ! -e $KNN_PCA_CORW_RESULT ]]; then
    mkdir -p $KNN_PCA_CORW_RESULT
    if ! train_knn_pca $KNN_PCA_CONF $KNN_PCA_CORW_RESULT; then
        cleanup $KNN_PCA_CORW_RESULT
	exit 10
    fi
fi

##################
## KNN ICA+CORW ##
##################

if [[ ! -e $KNN_ICA_CORW_RESULT ]]; then
    mkdir -p $KNN_ICA_CORW_RESULT
    if ! train_knn_ica $KNN_ICA_CONF $KNN_ICA_CORW_RESULT; then
        cleanup $KNN_ICA_CORW_RESULT
	exit 10
    fi
fi

##############
## KNN CORG ##
##############

if [[ ! -e $KNN_CORG_RESULT ]]; then
    mkdir -p $KNN_CORG_RESULT
    if ! train_knn_corg $KNN_CORG_CONF $KNN_CORG_RESULT; then
        cleanup $KNN_CORG_RESULT
	exit 10
    fi
fi

################
## KNN COVRED ##
################

if [[ ! -e $KNN_COVRED_RESULT ]]; then
    mkdir -p $KNN_COVRED_RESULT
    if ! train_knn_covred $KNN_COVRED_CONF $KNN_COVRED_RESULT; then
        cleanup $KNN_COVRED_RESULT
	exit 10
    fi
fi

##################
## BMC ENSEMBLE ##
##################

mkdir -p $BMC_ENSEMBLE_RESULT
if ! bmc_ensemble; then
    cleanup $BMC_ENSEMBLE_RESULT
    exit 10
fi

cp $BMC_ENSEMBLE_RESULT/test.txt $SUBMISSIONS_PATH/best_ensemble.txt
cp $ANN5_PCA_CORW_RESULT/test.txt $SUBMISSIONS_PATH/best_simple_model.txt

echo "The submissions are located at:"
echo "   - $SUBMISSIONS_PATH/best_ensemble.txt our best ensemble submission;"
echo "   - $SUBMISSIONS_PATH/best_simple_model.txt our best simple model submission."
