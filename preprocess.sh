#!/bin/bash
if [[ ! $(which april-ann) ]]; then
    echo "Needs APRIL-ANN toolkit, and april-ann command in the PATH"
    echo "https://github.com/pakozm/april-ann"
    exit -1
fi
. scripts/conf.txt
mkdir -p $TMP_PATH

if [[ ! -e $SEQUENCES_PATH ]]; then
    april-ann scripts/PREPROCESS/generate_sequences.lua
fi
