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

subjects <- c(unlist(strsplit(Sys.getenv("SUBJECTS"), " ")))
sources <- Sys.getenv("DATA_PATH")
files <- dir(sources)
destinationPath <- Sys.getenv("COVRED_PATH")

lo1perros<-23976

for (hh in 1:length(individuos)){
  indiv<-individuos[hh]
  files <- dir(paste(fuentes,indiv,sep=""))
  
  f<-files[1]

  mat <- readMat(paste(fuentes,indiv,"/",f,sep=""))[[1]]
  sampling.frequency <- mat[,,1]$sampling.frequency
  A <- mat[,,1]$data
  nchan <- dim(A)[1]
  lo <- dim(A)[2]
  lo1 <- as.integer(lo/10)
  rm(A)
  
  seginter <- grep("interictal",files)
  ninter <- length(seginter)
  
  segpre<-grep("preictal",files) #42
  npre<-length(segpre)
  
  segtest<-grep("test",files) #1000
  ntest<-length(segtest)
  
  ncoefs<-15 #Number of elements in Fourier basis

  A1<-array(dim=c(nchan,10,ninter+npre,ncoefs))
  A1.test<-array(dim=c(nchan,10,ntest,ncoefs))
  
  SDmatrix<-array(dim=c(nchan,ninter+npre))
  SDmatrix.test<-array(dim=c(nchan,ntest))
  
  ncoefs<-15 #Número de elementos en la base de Fourier

  #inter files
  for (i in 1:ninter){
    fich<-files[seginter[i]]
    Aux<- readMat(paste(fuentes,indiv,"/",fich,sep=""))[[1]][,,1]$data
    if (humanos[hh]=="S"){ #Re-sampling
      paso<-(1:239760)*as.integer(lo1/lo1perros)
      Aux<-Aux[,paso]
      lo1<-lo1perros
    }
    for (chan in 1:nchan){
      SDmatrix.test[chan,i]<-sd(Aux[chan,])
      for (t in 1:10){
        Aux1<-Aux[chan,(((t-1)*lo1)+1):(t*lo1)]
        Aux.coef<-t(fdata2fd(fdata(Aux1),type.basis="fourier",nbasis=15)$coefs)
        A1[chan,t,i,]<-Aux.coef
      }
    }
  }
  #pre files
  for (i in 1:npre){
    fich<-files[segpre[i]]
    Aux<- readMat(paste(fuentes,indiv,"/",fich,sep=""))[[1]][,,1]$data
    if (humanos[hh]=="S"){ #Re-sampling
      paso<-(1:239760)*as.integer(lo1/lo1perros)
      Aux<-Aux[,paso]
      lo1<-lo1perros
    }
    for (chan in 1:nchan){
      SDmatrix[chan,ninter+i]<-sd(Aux[chan,])
      for (t in 1:10){
        Aux1<-Aux[chan,(((t-1)*lo1)+1):(t*lo1)]
        Aux.coef<-t(fdata2fd(fdata(Aux1),type.basis="fourier",nbasis=15)$coefs)
        A1[chan,t,ninter+i,]<-Aux.coef
      }
    }
  }
  rm(Aux)
  rm(Aux1)
  rm(Aux.coef)
  
  #test files
  for (i in 1:ntest){
    fich<-files[segtest[i]]
    Aux<- readMat(paste(fuentes,indiv,"/",fich,sep=""))[[1]][,,1]$data
    if (humanos[hh]=="S"){ #Re-sampling
      paso<-(1:239760)*as.integer(lo1/lo1perros)
      Aux<-Aux[,paso]
      lo1<-lo1perros
    }
    for (chan in 1:nchan){
      SDmatrix.test[chan,i]<-sd(Aux[chan,])
      for (t in 1:10){
        Aux1<-Aux[chan,(((t-1)*lo1)+1):(t*lo1)]
        Aux.coef<-t(fdata2fd(fdata(Aux1),type.basis="fourier",nbasis=15)$coefs)
        A1.test[chan,t,i,]<-Aux.coef
      }
    }
  }
  rm(Aux)
  rm(Aux1)
  rm(Aux.coef)
  
  A1.sdcoefs<-apply(A1,c(1,2,3),sd)
  A1.sdcoefs.test<-apply(A1.test,c(1,2,3),sd)

  aux1<-apply(A1.sdcoefs,c(1,3),mean)
  aux1.test<-apply(A1.sdcoefs.test,c(1,3),mean)
  
  unedatos<-cbind(t(aux1),t(SDmatrix)) #covariates
  unedatosTEST<-cbind(t(aux1.test),t(SDmatrix.test)) #covariates
  
  for(i in 1:ninter){
    nombrefichero<-paste(substr(as.character(files[i]),1,nchar(as.character(files[i]))-4),"COVRED.txt",sep="")
    write.table(t(unedatos[i,]),file =nombrefichero, sep = " ", col.names = FALSE, row.names = FALSE)
  }
  for (i in 1:npre){
    nombrefichero<-paste(substr(as.character(files[(ninter+i)]),1,nchar(as.character(files[(ninter+i)]))-4),"COV.txt",sep="")
    write.table(unedatos[(ninter+i),],file =nombrefichero, sep = " ", col.names = FALSE, row.names = FALSE)
  }
  for (i in 1:ntest){
    nombrefichero<-paste(substr(as.character(files[(ninter+npre+i)]),1,nchar(as.character(files[(ninter+npre+i)]))-4),"COV.txt",sep="")
    write.table(t(unedatosTEST[i,]),file =nombrefichero, sep = " ", col.names = FALSE, row.names = FALSE)
  }
  cat(paste("Fin fichero ",hh,sep=""))
}  
