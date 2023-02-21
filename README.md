<<<<<<< HEAD
# Microsnoop

A generalist tool for the unbiased representation of heterogeneous microscopy images

### **Description**<img src="./logo.svg" width="280" title="scellseg" alt="microsnoop" align="right" vspace = "30">

Accurate and automated representation of microscopy images from small-scale to high-throughput is becoming an essential procedure in basic and applied biological research. Here, we present Microsnoop, a novel deep learning-based representation tool trained on large-scale microscopy images using masked self-supervised learning, which eliminates the need for manual annotation. Microsnoop is able to unbiasedly profile a wide range of complex and heterogeneous images, including single-cell, fully-imaged and batch-experiment data. We evaluated the performance of Microsnoop using seven high-quality datasets, containing over 358,000 images and 1,270,000 single cells with varying resolutions and channels from cellular organelles to tissues. Our results demonstrate Microsnoop's robustness and state-of-the-art performance in all biological applications, outperforming previous generalist and even custom algorithms. Furthermore, we presented its potential contribution for multi-modal studies. Microsnoop is highly inclusive of GPU and CPU capabilities, and can be freely and easily deployed on local or cloud computing platforms.

### Install

Operating system: It has been tested on Ubuntu. Theoretically, it can work on any system that can run Python.

Programing language: Python.

Our Environment: Python --3.7，CUDA --11.6， GPU --NVIDIA GeForce RTX 3090

This project uses h5py, numpy, opencv-python, scipy, pandas, kneed, faiss, tqdm, scikit-learn, torch, scellseg, . Go check them out if you don't have them, you can install them with conda or pip.

### How to use



### **Declaration**

Our pipeline referred to the following projects:
CytoImageNet: https://github.com/stan-hua/CytoImageNet
MAE: https://github.com/facebookresearch/mae
Scellseg: https://github.com/cellimnet/scellseg-publish
CPJUMP1: https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted