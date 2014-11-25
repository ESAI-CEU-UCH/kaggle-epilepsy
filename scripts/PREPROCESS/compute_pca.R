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

library(plyr)

sources <- Sys.getenv("FFT_PATH")
dest <- Sys.getenv("PCA_TRANS_PATH")

for(subject in c("Dog_1", "Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")){
    match <- paste("^", subject, "_(.*)ictal(.*)channel_(..).csv.gz$", sep="")
    list <- list.files(pattern = match, full.names=TRUE)
    data.list = ldply(list, function(path) { dum = read.table(path, header=F, sep=" ")})
    data.binded <- do.call(cbind, data.list)
    pca <- prcomp(data.binded, scale=TRUE, center=TRUE)
    output.rotation <- paste(dest, "/", subject, "_pca_rotation.txt", sep="")
    output.center <- paste(dest, "/", subject, "_pca_center.txt", sep="")
    output.scale <- paste(dest, "/", subject, "_pca_scale.txt", sep="")
    write.table(pca$rotation, file = output.rotation,
                sep = " ", col.names = FALSE, row.names = FALSE)
    write.table(pca$center, file = output.center,
                sep = " ", col.names = FALSE, row.names = FALSE)
    write.table(pca$scale, file = output.scale,
                sep = " ", col.names = FALSE, row.names = FALSE)
    rm(data)
}
