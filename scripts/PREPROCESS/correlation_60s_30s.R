library(R.matlab)
library(MASS)
sources <- Sys.getenv("DATA_PATH")
destinationPath <- Sys.getenv("WINDOWED_COR_PATH")

wsize <- 60
wadvance <- 30

for(subject in c("Dog_1", "Dog_2","Dog_3","Dog_4","Dog_5","Patient_1","Patient_2")){
    files <- dir(paste(sources,subject,sep="/"))
    for(f in files){
        dest.file <- paste(destinationPath,sub(".mat",".txt",f),sep="/")
        if(!file.exists(dest.file)) {
            A <- readMat(paste(sources,subject,f,sep="/"))[[1]][,,1]$data
            nchannels <- nrow(A)
            nsamples <- ncol(A)
            sampling.frequency <- readMat(paste(sources,subject,f,sep="/"))[[1]][,,1]$sampling.frequency
            ws = floor(wsize * sampling.frequency)
            wa = floor(wadvance * sampling.frequency)
            eig <- matrix(nrow=0,ncol=nchannels)
            for( t in seq(0,nsamples-ws,wa) ) {
                slice <- (t+1):(t+ws)
                S <- matrix(ncol=nchannels,nrow=nchannels)
                for(i in 1:nchannels){
                    for(j in 1:nchannels){
                        S[i,j] <- cor(A[i,slice],A[j,slice])
                    }
                }
                eig <- rbind(eig,eigen(S)$values)
            } # for every window
            write.matrix(eig,file=dest.file)
        } # if file not exists
    } # for each f in files
} # for each subject
