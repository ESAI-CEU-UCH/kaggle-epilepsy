# ESAI-CEU-UCH system for Seizure Prediction Challenge

This work presents the system proposed by Universidad CEU Cardenal Herrera
(ESAI-CEU-UCH) at Kaggle American Epilepsy Society Seizure Prediction
Challenge. The proposed system was positioned as **4th** at
[Kaggle competition](https://www.kaggle.com/c/seizure-prediction).

Different kind of input features (different preprocessing pipelines) and
different statistical models are being proposed. This diversity was motivated to
improve model combination result.

It is important to note that any of the proposed systems use test set for any
calibration or related purpose. The competition allow to do this model
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

The system is prepared to run in a Linux system with
[Ubuntu 14.04 LTS](http://www.ubuntu.com/), but it can be run in other Debian
based distributions, but not tested.

## Hardware

The minimum requirements for the correct execution of this software are:

- 6GB of RAM.
- 1.5GB of disk space.

The experimentation has been performed in a cluster of **three** computers
with same hardware configuration:

- Dell PowerEdge 210 II server rack.
- Intel Xeon E3-1220 v2 at 3.10GHz with 16GB RAM (4 cores).
- 2.6TB of NFS storage.

# How to generate the solution
