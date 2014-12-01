# ESAI-CEU-UCH system for Seizure Prediction Challenge

This work presents the system proposed by Universidad CEU Cardenal Herrera
(ESAI-CEU-UCH) at Kaggle American Epilepsy Society Seizure Prediction
Challenge. The proposed system was positioned as **4th** at
[Kaggle competition](https://www.kaggle.com/c/seizure-prediction).

Different kind of input features (different preprocessing pipelines) and
different statistical models are being proposed. This diversity was motivated to
improve model combination result.

It is important to note that **any** of the proposed systems use test set for
any calibration or related purpose. The competition allow to do this model
calibration using test set, but doing it will reduce the reproducibility of the
results in a real world implementation.

# Dependencies

## Software

This system uses the following open source software:

- [APRIL-ANN](https://github.com/pakozm/april-ann) toolkit v0.4.0. It is a
  toolkit for pattern recognition with Lua and C/C++ core. Because this tool is
  very new, the installation and configuration has been written in the pipeline.
- [R project](http://www.r-project.org/) v3.0.2. For statistical computing, a
  wide spread tool in Kaggle competitions. Packages R.matlab, MASS, fda.usc,
  fastICA, stringr and plyr are necessary to run the system.

The system is prepared to run in a Linux platform with
[Ubuntu 14.04 LTS](http://www.ubuntu.com/), but it could run in other Debian
based distributions, but not tested.

## Hardware

The minimum requirements for the correct execution of this software are:

- 6GB of RAM.
- 1.5GB of disk space.

The experimentation has been performed in a cluster of **three** computers
with same hardware configuration:

- Server rack Dell PowerEdge 210 II.
- Intel Xeon E3-1220 v2 at 3.10GHz with 16GB RAM (4 cores).
- 2.6TB of NFS storage.

# How to generate the solution

## General settings

The configuration of the input data, subjects, and other stuff is in
`settings.sh` script. The following environment variables indicate the location
of data and result folders:

- `DATA_PATH=DATA` indicates where the original data is. It will be organized in
  subfolders, one for each available subjects, and this subfolders will contain
  the corresponding MAT files.
- `TMP_PATH=TMP` indicates the folder for intermediate results, as feature
  extraction and models trainig.
- `SUBMISSIONS_PATH=SUBMISSIONS` indicates where the two selected submissions
  will be generated.

All other environment variables are computed depending in this three root paths.
Note that all the system must be executed being in the root path of the git
repository. The list of available subjects depends in the subfolders of
`DATA_PATH`.

## Recipe to reproduce the solution

It is possible to train and test the both selected submissions by executing
the script `train.sh`:

```
$ ./train.sh
```

It generates intermediate files in `$TMP_PATH` folder. First, all the proposed
features are generated to disk:

1. `$TMP_PATH/FFT_60s_30s_BFPLOS` contains FFT filtered features using 1 min. windows.
2. `$TMP_PATH/FFT_60s_30s_BFPLOS_PCA/` contains the PCA transformation of (1).
3. `$TMP_PATH/FFT_60s_30s_BFPLOS_ICA/` contains the ICA transformation of (1).
4. `$TMP_PATH/COR_60s_30s/` contains eigen values of windowed correlation matrices,
   using 1 min. windows over segments.
5. `$TMP_PATH/CORG/` contains eigen values of correlation matrices computed over the
   whole segment.
6. `$TMP_PATH/COVRED/` contains different global statistics computed for each segment.

Besides the features, PCA and ICA transformations are computed for each subject,
and the transformation matrices are stored at:

- `$TMP_PATH/PCA_TRANS`
- `$TMP_PATH/ICA_TRANS`

This preprocess can be executed without training by using the script
`preprocess.sh`.

Once preprocessing step is ready, training of the proposed models starts. The
model results are stored in subfolders of `$TMP_PATH`. This subfolders contain
similar data:

- `validation_SUBJECT.txt` is the concatenation of cross-validation output, used
  to optimize the final ensemble.
- `validation_SUBJECT.test.txt` is the test results corresponding to the
  indicated subject (without CSV header).
- `test.txt` is the concatenation of all test results with the CSV header needed
  to send it as submission to Kaggle.

The trained systems are stored at folders:

1. `$TMP_PATH/ANN2P_ICA_CORW_RESULT/`
2. `$TMP_PATH/ANN5_PCA_CORW_RESULT/`
3. `$TMP_PATH/ANN2_ICA_CORW_RESULT/`
4. `$TMP_PATH/KNN_PCA_CORW_RESULT/`
5. `$TMP_PATH/KNN_ICA_CORW_RESULT/`
6. `$TMP_PATH/KNN_CORG_RESULT/`
7. `$TMP_PATH/KNN_COVRED_RESULT/`

Testing procedure is incorporated in training scripts, but it is possible to
run it using the script `test.sh`.

## Recipe to train a new subject

In order to train a new subject, you can use `train_subject.sh` script, which
receives as argument the name of the subject:

```
$ ./train_subject.sh SUBJECT_NAME
```


## Recipe to test new data for a trained subject
