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

library(R.matlab)
library(MASS)

subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("DATA_PATH")
destinationPath <- Sys.getenv("WINDOWED_COR_PATH")

wsize <- 60
wadvance <- 30

for(subject in subjects) {
    write(paste("#",subject), stdout())
    files <- dir(paste(sources,subject,sep="/"))
    for(f in files){
        dest.file <- paste(destinationPath,sub(".mat",".txt",f),sep="/")
        if(!file.exists(dest.file)) {
            mat <- readMat(paste(sources,subject,f,sep="/"))[[1]]
            A <- mat[,,1]$data
            nchannels <- nrow(A)
            nsamples <- ncol(A)
            sampling.frequency <- mat[,,1]$sampling.frequency
            ws = floor(wsize * sampling.frequency)
            wa = floor(wadvance * sampling.frequency)
            eig <- matrix(nrow=0,ncol=nchannels)
            for( t in seq(0,nsamples-ws,wa) ) {
                slice <- (t+1):(t+ws)
                eig <- rbind(eig,eigen(cor(t(A[,slice])))$values)
            } # for every window
            write.matrix(eig,file=dest.file)
            rm(A)
        } # if file not exists
    } # for each f in files
} # for each subject
