list.of.packages <- c("fastICA", "MASS", "R.matlab", "stringr", "plyr", "fda.usc", "foreach", "doMC")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) {
    install.packages(new.packages, repos='http://cran.us.r-project.org')
}
