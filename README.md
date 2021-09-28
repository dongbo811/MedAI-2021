# MedAI-2021

## Installation
Our model has been tested on Linux (Ubuntu 16, 18 and 20; centOS, RHEL). We do not provide support for other operating systems.

Our model requires a GPU! For inference, the GPU should have 6 GB of VRAM. For training, we recommend a strong CPU to go along with the GPU. At least 6 CPU cores (12 threads) are recommended, and the GPU should have at least 16 GB (popular non-datacenter options are the Tesla P100  or RTX 3090). 

We very strongly recommend you install our model in a virtual environment. Here is a quick how-to for Ubuntu. If you choose to compile pytorch from source, you will need to use conda instead of pip. In that case, please set the environment variable OMP\_NUM\_THREADS=1 (preferably in your bashrc using export OMP\_NUM\_THREADS=1). This is important!

Python 2 is deprecated and not supported. Please make sure you are using Python 3. Install PyTorch. You need at least version 1.6. Install our model depending on your use case.

### Recommended environment
```
Python 3.8
Pytorch 1.7.1
torchvision 0.8.2
```

### Clone
```
git clone https://github.com/dongbo811/MedAI-2021.git
cd MedAI-2021
```

## Data preparation
The development dataset for the instrument segmentation task can be downloaded via: {https://datasets.simula.no/kvasir-instrument/}. Then extract the archive to a destination of your choice. 

Use the link below to access and download the development dataset for the polyp segmentation task: {https://datasets.simula.no/kvasir-seg/}.  

If you want to use 5-fold cross-validation, you need to divide it into 5 different training sets and test sets. Here, the data structures are shown in follows.

```
5-fold cross-validation
1_test, 1_train
2_test, 2_train
3_test, 3_train
4_test, 4_train
5_test, 5_train

data/*_train/images/*.jpg
data/*_train/masks/*.jpg
data/*_test/images/*.jpg
data/*_test/masks/*.jpg
```
## Training
```
% training in a way of 5-fold cross-validation 
Cd Polyp-PVT
Python train.py
% training in a way of 5-fold cross-validation 
Cd Sinv2-PVT
Python train.py
% training in a way of 5-fold cross-validation 
Cd Transfuse-PVT
Python train.py
```
## Inference with trained models
```
% testingwith 5 different trained models 
Cd Polyp-PVT
Python test.py
% testingwith 5 different trained models  
Cd Sinv2-PVT
Python test.py
% testingwith 5 different trained models 
Cd Transfuse-PVT
Python test.py
```
