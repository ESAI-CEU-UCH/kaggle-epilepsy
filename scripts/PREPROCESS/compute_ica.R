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

library(fastICA)
library(stringr)

subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("FFT_PATH")
files <- dir(sources)
destinationPath <- Sys.getenv("FFT_ICA_PATH")

getChannelNumber <- function(path) {
    as.numeric(gsub("^.*channel_(..).*$", "\\1", path))
}

processSet <- function(auxfiles, seg, ninter, nchannels, data, start) {
    for (i in 1:ninter){
        for (j in 1:nchannels){
            path <- paste(sources, auxfiles[seg[(i-1)*nchannels+j]], sep="/")
            chan <- getChannelNumber(path)
            tt <- read.table(path)
            data[i+start,,,chan] <- as.matrix(tt)
        }
    }
    data
}

for (hh in 1:length(subjects)){
    subject <- subjects[hh]
    write(paste("#",subject), stdout())
    auxfiles <- files[grep(subject,files)]

    seginter <- grep("interictal",auxfiles)
    segpre <- grep("preictal",auxfiles)
    segtest <- grep("test",auxfiles)

    nchannels <- length(seginter) / length(grep("channel_01",
                                                auxfiles[seginter]))
    
    ninter <- length(seginter)/nchannels 
    npre <- length(segpre)/nchannels 
    ntest <- length(segtest)/nchannels
  
    tt <- read.table(paste(sources, auxfiles[seginter[1]], sep="/"))
    nfft <- dim(tt)[1]
    nfilt <- dim(tt)[2]

    # Read all training data
    data <- array(dim=c(ninter+npre,nfft,nfilt,nchannels))
    
    data <- processSet(auxfiles, seginter, ninter, nchannels, data, 0)
    data <- processSet(auxfiles, segpre, npre, nchannels, data, ninter)
    
    # Read test data
    datatest <- array(dim=c(ntest,nfft,nfilt,nchannels))

    datatest <- processSet(auxfiles, segtest, ntest, nchannels, datatest, 0)
        
    # Decompose and align by minute all the data
    data1 <- (apply(apply(data,c(2,1),unlist),1,unlist))
    datatest1 <- (apply(apply(datatest,c(2,1),unlist),1,unlist))

    # Compute ICA and process test data
    ica <- fastICA(data1,n.comp=dim(data1)[2],method="C")
    testica <- scale(datatest1,center=TRUE,scale=FALSE)%*%ica$K%*%ica$W

    for(i in 1:ninter){
        filename <- paste(destinationPath, "/",
                          gsub(".channel_.*$", "", auxfiles[i]), ".txt", sep="")
        write.table(ica$S[((i-1)*nfft+1):(i*nfft),],file =filename, sep = " ",
                    col.names = FALSE, row.names = FALSE)
    }
    for (i in 1:npre){
        filename <- paste(destinationPath, "/",
                          gsub(".channel_.*$", "", auxfiles[ninter + i]), ".txt", sep="")
        write.table(ica$S[((ninter + i-1)*nfft+1):((ninter + i)*nfft),],file =filename, sep = " ",
                    col.names = FALSE, row.names = FALSE)
    }
    
    for (i in 1:ntest){
        filename <- paste(destinationPath, "/",
                          gsub(".channel_.*$", "", auxfiles[ninter+npre+i]), ".txt", sep="")
        write.table(testica[((i-1)*nfft+1):(i*nfft),],file =filename, sep = " ",
                    col.names = FALSE, row.names = FALSE)
    }
}
