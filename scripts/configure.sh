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

# check competition data is in the proper path
if [[ ! -e DATA/Dog_1 || ! -e DATA/Dog_2 || ! -e DATA/Dog_3 || ! -e DATA/Dog_4 || ! -e DATA/Dog_5 || ! -e DATA/Patient_1 || ! -e DATA/Patient_2 ]]; then
    echo "ERROR: Please, download and uncompress all competition data into $ROOT_PATH/DATA"
    echo "       or just make a symbolic link to directory where data is stored"
    exit 10
fi

VERSION=0.4.0
APRILANN=april-ann-$VERSION

cleanup()
{
    echo "CLEANING UP, PLEASE WAIT UNTIL FINISH"
    cd $ROOT_PATH
    rm -Rf $TMP_PATH/$APRILANN
    md5sum --quiet -c $ROOT_PATH/scripts/v0.4.0.md5
    err=$?
    if [[ $? -ne 0 ]]; then
        rm -f $TMP_PATH/v$VERSION.tar.gz
    fi
}
 
# control-c execution
control_c()
{
    echo -en "\n*** Exiting by control-c ***\n"
    cleanup
    exit 10
}
 
# trap keyboard interrupt (control-c)
trap control_c SIGINT

# check that APRIL_EXEC exists and links to a valid file
if [[ -z $APRIL_EXEC || $APRIL_EXEC = "" || ! -e $APRIL_EXEC ]]; then
    # check if april-ann directory exists
    if [[ ! -e $TMP_PATH/$APRILANN ]]; then
        cd $TMP_PATH
        # check if source code has been downloaded
        if [[ ! -e v0.4.0.tar.gz ]]; then
            echo "Downloading APRIL-ANN tarball"
            wget https://github.com/pakozm/april-ann/archive/v0.4.0.tar.gz
            md5sum --quiet -c $ROOT_PATH/scripts/v0.4.0.md5
            err=$?
            if [[ $? -ne 0 ]]; then
                echo "ERROR: Unable to check md5sum of downloaded APRIL-ANN tarball"
                cleanup
                exit 10
            fi
        fi
        echo "Unpacking APRIL-ANN tarball"
        if ! tar zxf v0.4.0.tar.gz; then
            echo "ERROR: Unable to unpack APRIL-ANN tarball"
            cleanup
            exit 10
        fi
        echo "Instaling dependencies and compiling APRIL-ANN"
        cd $APRILANN
        # compilation process, if any error happens, the whole directory will be
        # removed
        echo "You will need to be sudoer for properly install dependencies"
        if ! ./DEPENDENCIES-INSTALLER.sh; then
            echo "ERROR: Unable to install dependencies"
            cleanup
            exit 10
        fi
        . configure.sh
        if ! make release-atlas; then
            echo "ERROR: Unable to compile APRIL-ANN :'("
            cleanup
            exit 10
        fi
        if ! make test; then
            echo "ERROR: Unable to test APRIL-ANN :'("
            cleanup
            exit 10
        fi
        echo "APRIL-ANN installed, compiled and tested correctly :-)"
    fi
    # configure APRIL-ANN to export APRIL_EXEC environment variable
    echo "Configuring APRIL-ANN"
    cd $ROOT_PATH/$TMP_PATH/$APRILANN &&
    . configure.sh
    if [[ $? -ne 0 ]]; then
        cd $ROOT_PATH
        echo "ERROR: Unable to configure APRIL-ANN :'("
        exit 10
    fi
fi
# removes keyboard interrupt trap (control-c)
trap - SIGINT

echo "APRIL-ANN configured properly :-)"
cd $ROOT_PATH
