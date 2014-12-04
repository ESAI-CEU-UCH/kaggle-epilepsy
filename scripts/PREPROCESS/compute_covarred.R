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
library(fda.usc)
library(foreach)
library(doMC)

registerDoMC(Sys.getenv("OMP_NUM_THREADS"))  # the number of CPU cores
subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("DATA_PATH")
destinationPath <- Sys.getenv("COVRED_PATH")

#freqbase <- 400
ncoefs <- 15 # Number of elements in Fourier basis
nslices <- 10
lbase <- 23976
lbase10 <- 239760

for (subject in subjects) {
    write(paste("#",subject), stdout())
    files <- dir(paste(sources,subject,sep="/"))
    foreach(i=1:length(files)) %dopar% {
        f <- files[i]
        outname <- paste(destinationPath, "/", substr(f,1,nchar(f)-4), ".txt", sep="")
        if(!file.exists(outname)){
            mat <- readMat(paste(sources,subject,f,sep="/"))[[1]]
            sampling.frequency <- round(mat[,,1]$sampling.frequency)
            data <- mat[,,1]$data
            nchan <- dim(data)[1]
            L <- dim(data)[2]
            A1 <- array(dim=c(nchan,nslices,ncoefs))
            SDmatrix <- array(dim=c(nchan))
            lo1 <- as.integer(L / nslices)
            Aux <- data
            if (lo1 != lbase) {
                steps <- (1:lbase10)*as.integer(lo1/lbase)
                # steps <- seq(from=1, to=L, by=as.integer(sampling.frequency/freqbase))
                Aux <- data[,steps]
            }
            for (chan in 1:nchan) {
                SDmatrix[chan] <- sd(data[chan,]) # without resampling
                for (t in 1:nslices){
                    Aux1 <- Aux[chan,(((t-1)*lbase)+1):(t*lbase)]
                    Aux.coef <- t(fdata2fd(fdata(Aux1),type.basis="fourier",nbasis=ncoefs)$coefs)
                    A1[chan,t,] <- Aux.coef
                }
            }
            A1.sdcoefs <- apply(A1, c(1,2), sd)
            aux1 <- apply(A1.sdcoefs, 1, mean)
            result <- cbind(t(aux1), t(SDmatrix))
            write.table(result, file=outname, sep=" ",
                        col.names=FALSE, row.names=FALSE)
            rm(mat)
            rm(data)
            rm(A1)
            rm(SDmatrix)
            rm(Aux)
            rm(result)
        }
    }
}
