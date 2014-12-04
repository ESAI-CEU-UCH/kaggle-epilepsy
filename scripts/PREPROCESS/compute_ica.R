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

SEED <- 37442
subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("FFT_PATH")
files <- dir(sources)
dest <- Sys.getenv("ICA_TRANS_PATH")

getChannelNumber <- function(path) {
    as.numeric(gsub("^.*channel_(..).*$", "\\1", path))
}

readSet <- function(auxfiles, seg, nlen, nfft, nfilt, nchannels) {
    data <- array(dim=c(nlen,nfft,nfilt,nchannels))
    for (i in 1:nlen){
        for (j in 1:nchannels){
            path <- paste(sources, auxfiles[seg[(i-1)*nchannels+j]], sep="/")
            chan <- getChannelNumber(path)
            tt <- read.table(path)
            data[i,,,chan] <- as.matrix(tt)
        }
    }
    data
}

for (hh in 1:length(subjects)){
    subject <- subjects[hh]
    write(paste("#",subject), stdout())
    output.center <- paste(dest, "/", subject, "_ica_center.txt", sep="")
    output.center2 <- paste(dest, "/", subject, "_ica_center2.txt", sep="")
    output.K <- paste(dest, "/", subject, "_ica_K.txt", sep="")
    output.W <- paste(dest, "/", subject, "_ica_W.txt", sep="")
    if (!file.exists(output.center) || !file.exists(output.K) ||
        !file.exists(output.W) || !file.exists(output.center2) ) {
        auxfiles <- files[grep(subject,files)]

        trseg <- grep("ictal",auxfiles)
        
        nchannels <- length(trseg) / length(grep("channel_01",
                                                 auxfiles[trseg]))
        
        ntraining <- length(trseg)/nchannels 
        
        tt <- read.table(paste(sources, auxfiles[trseg[1]], sep="/"))
        nfft <- dim(tt)[1]
        nfilt <- dim(tt)[2]
        
                                        # Read all training data
        data <- readSet(auxfiles, trseg, ntraining, nfft, nfilt, nchannels)
        
                                        # Decompose and align by minute all the data
        data1 <- (apply(apply(data,c(2,1),unlist),1,unlist))

                                        # Compute centers
        center <- colMeans(data1)
        
                                        # Compute ICA
        set.seed(SEED)
        ica <- fastICA(data1,n.comp=dim(data1)[2],method="C")

                                        # testica <- scale(datatest1,center=center,scale=FALSE)%*%ica$K%*%ica$W

                                        # Write transformation matrices
        write.table(center, file = output.center,
                    sep = " ", col.names = FALSE, row.names = FALSE)
        write.table(ica$K, file = output.K,
                    sep = " ", col.names = FALSE, row.names = FALSE)
        write.table(ica$W, file = output.W,
                    sep = " ", col.names = FALSE, row.names = FALSE)

        # BUG: we used centers of test instead of training in Kaggle submission
        testseg <- grep("test",auxfiles)
        data2 <- readSet(auxfiles, testseg, length(testseg)/nchannels, nfft, nfilt, nchannels)
        center2 <- colMeans( (apply(apply(data2,c(2,1),unlist),1,unlist)) )
        write.table(center2, file = output.center2,
                    sep = " ", col.names = FALSE, row.names = FALSE)        
    }
}
