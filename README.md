# Introduction

This work presents the system proposed by Universidad CEU Cardenal
Herrera (ESAI-CEU-UCH) at Kaggle challenge in seizure epilepsy
prediction. The proposed system was positioned as **4th** at Kaggle
(https://www.kaggle.com/c/seizure-prediction).

Different kind of input features (different preprocessing
pipelines) and different statistical models are being proposed. This diversity
was motivated to improve model combination result.

# Preprocessing

Several preprocessing pipelines has been performed, being the most important the
FFT plus PCA step.

## FFT features plus PCA

This is the most important kind of proposed features, obtaining the best
standalone result. The process follow this steps for each input file:

1. Extract windows of 60 seconds size with 50% overlapping for every channel in
   the data, increasing the number of samples in the data set. We obtained 19
   windows for every input file. Every sample is filtered by using Hamming
   windows.
2. For every window, real FFT is computed using an algorithm which needs power
   of two window sizes, so, 60s input windows are of 24000 for Dogs and 300000
   for Patients, leading into 16384 FFT bins for Dogs and 262144 FFT for
   Patients.
3. A filter bank is computed for bands: delta (0.1Hz - 4Hz), theta (4Hz - 8Hz),
   alpha (8Hz - 12Hz), beta (12Hz - 30Hz), low-gamma (30Hz - 70Hz), high-gamma
   (70Hz - 180Hz). The mean in the corresponding frequency bands has been
   computed.

For every file, it is computed a matrix with 19 rows (windows of 60s with 50%
overlapping) and 6*CHANNELS columns. This FFT transformation has been applied
for all the available data, training and test.

A PCA transformation has been computed over the concatenation by rows of all
training data matrices, leaving test data hidden for this step. Data has been
centered and scaled before PCA. After computing centers, scales and PCA matrix
transformation, it has been applied to all the available data, training and
test.

## Eigen values of pairwise channels correlation matrix

### Windowed correlation matrix

### Global correlation matrix

## Global statistical features

# Models

Three different models has been tested. Logistic regression as initial baseline
following a simple linear regression approach. KNNs for a more sophisticated
model. And finally dropout ANNs with different number of layers and ReLUs as
neurons.

## Logistic regression

## K-Nearest Neighbors

## Artificial Neural Networks

# Results

# Conclusions

# Software and hardware settings

## Software

This system uses the following open source software:

- [APRIL-ANN](https://github.com/pakozm/april-ann) toolkit. It is a toolkit for
  pattern recognition with Lua and C/C++ core. Because this tool is very new,
  the installation and configuration has been written in the pipeline.
- [R project](http://www.r-project.org/). For statistical computing, a wide
  spread tool in Kaggle competitions.

The system is prepared to run in a Linux system with
[Ubuntu 14.04 LTS](http://www.ubuntu.com/), but it can be run in other Debian
based distributions, but not tested.

## Hardware

The experimentation has been performed over a cluster of three computers with
same software configuration and the following hardware description:

- Dell PowerEdge 210 II server rack.
- Intel Xeon E3-1220 v2 at 3.10GHz with 16GB RAM.
- 2.6TB of NFS storage.
