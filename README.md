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
standalone result. The process follow this steps:

1. Extract windows of 60 seconds size with 50% overlapping for every channel in
   the data, increasing the number of samples in the data set. We obtained 19
   windows for every input file. Every sample is filtered by using Hamming
   windows.
2. For every window, real FFT is computed using an algorithm which needs power
   of two window sizes, so, 60s input windows are of 24000 for Dogs and 300000
   for Patients, leading into 16384 FFT bins for Dogs and 262144 FFT for
   Patients.
3. A filter bank is computed for bands delta (0.1 - 4 Hz), theta (

## Eigen values of pairwise channels correlation matrix

### Windowed correlation matrix

### Global correlation matrix

## Global statistical features

# Models

## Artificial Neural Networks

## K-Nearest Neighbors

# Results

# Conclusions
