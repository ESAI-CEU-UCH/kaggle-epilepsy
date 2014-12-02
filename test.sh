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

MLP_TEST_SCRIPT=scripts/MODELS/test_one_subject_mlp.lua
KNN_TEST_SCRIPT=scripts/MODELS/test_one_subject_knn.lua

TEST_OUTPUT=$(mktemp --tmpdir=$SUBMISSIONS_PATH/ $test.$subject.XXXXXX.txt)
BASE=$(basename $TEST_OUTPUT)

###############################################################################

cleanup()
{
    rm -f $TEST_OUTPUT
}

# control-c execution
control_c()
{
    echo -en "\n*** Exiting by control-c ***\n"
    cleanup
    exit 10
}

test_subject()
{
    script=$1
    model=$2
    subject=$3
    mkdir -p $model.TEST
    $APRIL_EXEC $script $model $subject $model.TEST/validation_$subject.test.txt
    return $?
}

test_mlp()
{
    $model=$1
    $subject=$2
    test_subject $MLP_TEST_SCRIPT $model $subject
    return $?
}

test_knn()
{
    $model=$1
    $subject=$2
    test_subject $KNN_TEST_SCRIPT $model $subject
    return $?
}

bmc_ensemble()
{
    $APRIL_EXEC scripts/ENSEMBLE/bmc_ensemble.lua \
        $ANN2P_PCA_CORW_RESULT.TEST $ANN5_PCA_CORW_RESULT.TEST $ANN2_ICA_CORW_RESULT.TEST \
        $KNN_ICA_CORW_RESULT.TEST $KNN_PCA_CORW_RESULT.TEST \
        $KNN_CORG_RESULT.TEST $KNN_COVRED_RESULT.TEST > $1
    return $?
}

###############################################################################

if [[ ( ! -e $FFT_PCA_PATH ) || ( ! -e $FFT_ICA_PATH ) || ( ! -e $WINDOWED_COR_PATH ) || ( ! -e $CORG_PATH ) || ( ! -e $COVRED_PATH ) ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $ANN5_PCA_CORW_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $ANN2P_PCA_CORW_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $ANN2_ICA_CORW_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $KNN_PCA_CORW_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $KNN_ICA_CORW_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $KNN_CORG_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

if [[ ! -e $KNN_COVRED_RESULT ]]; then
    echo "Execute train.sh before test.sh"
    exit 10
fi

#############################################################################

for $subject in $SUBJECTS; do
    echo "# $subject"
    
    if ! test_mlp $ANN5_PCA_CORW_RESULT $subject; then
        cleanup
        exit 10
    fi

    if ! test_mlp $ANN2P_PCA_CORW_RESULT; then
        cleanup
        exit 10
    fi
    
    if ! test_mlp $ANN2_ICA_CORW_RESULT; then
        cleanup
        exit 10
    fi

    if ! test_knn $KNN_PCA_CORW_RESULT; then
        cleanup
        exit 10
    fi

    if ! test_knn $KNN_ICA_CORW_RESULT; then
        cleanup
        exit 10
    fi

    if ! test_knn $KNN_CORG_RESULT; then
        cleanup
        exit 10
    fi

    if ! test_knn $KNN_COVRED_RESULT; then
        cleanup
        exit 10
    fi
    
done

if ! bmc_ensemble; then
    cleanup
    exit 10
fi

echo "The result is located at:"
echo "   - $test_output our best ensemble submission."
