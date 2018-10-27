# German Traffic Sign Recognition Challenge using ResNets

This repository contains a simple, light and high accuracy model for the German Traffic Sign Recognition Benchmark ([GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)) dataset. This model was designed and trained for the [NYU's Fall 2018 Computer Vision course competition in Kaggle](https://www.kaggle.com/c/nyu-cv-fall-2018). All training was done using GPUs in NYU's Prince cluster.

The baseline code for training and producing predictions was obtained [here](https://github.com/soumith/traffic-sign-detection-homework) and modified in this repository.

The Residual Network implemented in `model48.py` obtained 99.02% accuracy as single model in the test set of GTSRB. Although not being the highest score of this dataset, this model can easily be trained in less than 30 minutes in a single GPU.

