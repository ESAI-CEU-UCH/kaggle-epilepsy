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
files <- dir(sources)
destinationPath <- Sys.getenv("CORG_PATH")

corrDiffAv <- function(subject,f){
    outname <- paste(destinationPath,"/",substr(f,1,nchar(f)-4),".txt",sep="")
    if(!file.exists(outname)){
        mat <- readMat(paste(sources,subject,f,sep="/"))[[1]]
        A <- mat[,,1]$data
        nchannels <- nrow(A)
        n <- ncol(A)
        sampling.frequency <- mat[,,1]$sampling.frequency
        step <- 1
        Adiff <- matrix(nrow=nchannels,ncol=n-step)
        for(k in 1:nchannels){
            Adiff[k,] <- A[k,(1+step):n]-A[k,1:(n-step)]
        }
        CorrA <- cor(t(data.matrix(A)))
        CorrAdiff<-cor(t(data.matrix(Adiff)))
        eigCorrA <- matrix(eigen(CorrA)$values,ncol=nchannels,nrow=1)
        eigCorrAdiff <- matrix(eigen(CorrAdiff)$values,ncol=nchannels,nrow=1)
        write.matrix(cbind(eigCorrA,eigCorrAdiff), outname)
        #
        rm(A)
        rm(mat)
        rm(Adiff)
        rm(CorrA)
        rm(CorrAdiff)
    }
}

for(subject in subjects) {
    write(paste("#",subject), stdout())
    for(filename in dir(paste(sources, subject, sep="/"))) {
        corrDiffAv(subject, filename)
    }
}
