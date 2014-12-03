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

readMatrix <- function(path) { read.table(path, header=F, sep=" ") }

subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("FFT_PATH")
dest <- Sys.getenv("PCA_TRANS_PATH")

for (subject in subjects) {
    write(paste("#",subject), stdout())
    output.rotation <- paste(dest, "/", subject, "_pca_rotation.txt", sep="")
    output.center <- paste(dest, "/", subject, "_pca_center.txt", sep="")
    output.scale <- paste(dest, "/", subject, "_pca_scale.txt", sep="")
    if (!file.exists(output.rotation) || !file.exists(output.center) || !file.exists(output.scale)) {
        i=0
        cols = list()
        while(TRUE) {
            i=i+1
            match <- sprintf("^%s_(.*)ictal(.*)channel_%02d.csv.gz$", subject, i)
            list <- list.files(path = sources, pattern = match, full.names=TRUE)
            if (length(list) == 0) break;
            cols[[i]] <- do.call(cbind, ldply(list, readMatrix))
        }
        data <- do.call(cbind, cols)
        rm(cols)
        pca <- prcomp(data, scale=TRUE, center=TRUE)
        write.table(pca$rotation, file = output.rotation,
                    sep = " ", col.names = FALSE, row.names = FALSE)
        write.table(pca$center, file = output.center,
                    sep = " ", col.names = FALSE, row.names = FALSE)
        write.table(pca$scale, file = output.scale,
                    sep = " ", col.names = FALSE, row.names = FALSE)
        rm(data)
    }
}
