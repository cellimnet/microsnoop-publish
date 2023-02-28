# Microsnoop

A generalist tool for the unbiased representation of heterogeneous microscopy images

### **Description**<img src="./logo.svg" width="280" title="Microsnoop" alt="Microsnoop" align="right" vspace = "10">

Automated and accurate profiling of microscopy images from small-scale to high-throughput is becoming an essential procedure in basic and applied biological research. Here, we present Microsnoop, a novel deep learning-based representation tool trained on large-scale microscopy images using masked self-supervised learning, which eliminates the need for manual annotation. Microsnoop is able to unbiasedly profile a wide range of complex and heterogeneous images, including single-cell, fully-imaged and batch-experiment data. We evaluated the performance of Microsnoop using seven high-quality datasets, containing over 358,000 images and 1,270,000 single cells with varying resolutions and channels from cellular organelles to tissues. Our results demonstrate Microsnoop's robustness and state-of-the-art performance in all biological applications, outperforming previous generalist and even custom algorithms. Furthermore, we presented its potential contribution for multi-modal studies. Microsnoop is highly inclusive of GPU and CPU capabilities, and can be freely and easily deployed on local or cloud computing platforms.

#### Overview of Microsnoop<img src="./overview.png" width="850" title="Overview of Microsnoop" alt="Overview of Microsnoop" align="left">



































**Fig. 1 | Design of Microsnoop for microscopy image representation. a,** Schematic of the learning process. (i) Example of the four main category images are shown. The channels range from cellular organelles to tissues. (ii) A masked self-supervised learning strategy was employed and only images are required for training without additional manual annotation. One-channel masked images were set as the input and the Encoder- Decoder were required to reconstruct the original images. **b,** At test time, (i) Example images from various downstream tasks are shown, with different resolutions, number of channels and image types. These microscopy images are categorized into 3 types to ensure the broad coverage of image profiling needs. (ii) Application of Microsnoop. Each batch of images is fed into the pre-trained encoder, and the output smallest convolutional maps are processed by average pooling. Then, all extracted embeddings are processed according to different profiling tasks. The potential downstream analyses of our generalist representation tool are shown in the panel.

### Install

Operating system: It has been tested on Ubuntu. Theoretically, it can work on any system that can run Python.

Programing language: Python.

Our Environment: Python --3.7，CUDA --11.6， GPU --NVIDIA GeForce RTX 3090

This project uses h5py, numpy, opencv-python, scipy, pandas, kneed, faiss, tqdm, scikit-learn, torch, scellseg. Go check them out if you don't have them, you can install them with conda or pip.

#### Amazon Cloud Computing

A configured Amazon Machine Image (AMI) is available at Community AMIs. You can follow the following steps to quickly deploy Microsnoop for microscopy image analysis.

1. Launch instance from AMI: search and choose our AMI --- Microsnoop-publish-20230228
2. Choose suitable hardware, e.g. CPU, GPU, storage
3. Configure SSH: the name of our configured env is pytorch_latest_p37
4. Map your local project to the deployment project

### Usage

We provide examples of using Microsnoop for profiling single-cell (run_cyclops), fully-imaged (run_livecell), and batch-experiment  (run_bbbc021) images, corresponding data can be obtained at https://microsnoop-publish-data.s3.cn-northwest-1.amazonaws.com.cn/evaluation_datasets/. You can follow these examples to build your own process. Any questions on the use of the software can be contacted via Issues and we will reply promptly.

### **Reference**

Our pipeline referred to the following projects:

CytoImageNet: https://github.com/stan-hua/CytoImageNet

MAE: https://github.com/facebookresearch/mae

Scellseg: https://github.com/cellimnet/scellseg-publish

CPJUMP1: https://github.com/jump-cellpainting/2021_Chandrasekaran_submitted

Cellpose: https://github.com/MouseLand/cellpose