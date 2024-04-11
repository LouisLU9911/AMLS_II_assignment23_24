# AMLS_II_assignment23_24

* Name: Zhaoyan LU
* Student ID: 23049710

## Contents

- [Overview](#Overview)
- [Repo Structure](#Repo-Structure)
- [Requirements](#Requirements)
- [Datasets](#Datasets)
- [Usage](#Usage)

## Overview

This repo is for the final assignment of ELEC0135 Applied Machine Learning Systems II (23/24).
This project aims at solving the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/) task
using fine-tuned pre-trained models or Mixtures of Experts (MoE).

## Repo Structure

```bash
$ tree
.
├── A
│   ├── __init__.py
│   ├── config.json                             # training config
│   ├── confusion_matrix_of_base_model.png
│   ├── constants.py
│   ├── dataset.py                              # load datasets
│   ├── draw_confusion_matrix.py
│   ├── draw_learning_curves.py
│   ├── learning_curves.png
│   ├── logger.py
│   ├── model                                   # core module of Mixtures of Experts
│   │   ├── __init__.py
│   │   ├── configuration_moe.py                # MoE model configurations
│   │   └── modeling_moe.py                     # MoE model; See https://huggingface.co/docs/transformers/custom_models#sharing-custom-models
│   ├── moe.py                                  # entrypoint of moe action; train and test
│   ├── pretrained.py                           # entrypoint of pretrained action; train and test
│   ├── y.txt
│   └── y_pred.txt
├── Datasets
│   └── prepare.py                              # format dataset to the imagefolder format; See https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder
├── Makefile
├── README.md
├── environment.yml
├── main.py                                     # entrypoint of the assignment
└── requirements.txt
```

## Requirements

* Miniconda / Anaconda
* Nvidia GPU with cuda 11.8

* `environment.yml`

```bash
$ cat environment.yml
name: amls_ii-final-uceezl8
channels:
  - defaults
dependencies:
  - python=3.9
  - black                               # formatter
  - scikit-learn                        # confusion matrix
  - pandas                              # load dataframe
  - ipython                             # debugging
  - pillow                              # read images
  - torchvision                         # pre-processing
  - pytorch                             # build models
  - pytorch-cuda=11.8                   # pytorch for GPU
  - matplotlib                          # draw figures
  - pip:
    - -r requirements.txt
```

* `requirements.txt`

```bash
$ cat requirements.txt
# download the dataset from kaggle
kaggle
# preprocess
datasets
# metrics
evaluate
# fine-tuning and build custom models
transformers[torch]==4.39.3
ipykernel
```

Create conda env:

```bash
$ make create-env
# or
$ conda env create -f environment.yml
```

## Datasets

We are using the dataset provided by the [competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data).
If you want to download the dataset, you may need to join this competition first.

```bash
# Step 0: enter conda env
$ conda activate amls_ii-final-uceezl8
# Step 1: download the dataset and extract it
$ make dataset
# or
$ cd Datasets
$ kaggle competitions download -c cassava-leaf-disease-classification
$ unzip -q cassava-leaf-disease-classification.zip
$ python prepare.py
```

## Usage

* info

```bash
$ python main.py info
-------------------------------------
|      AMLS II Assignment 23-24     |
|         Name: Zhaoyan Lu          |
|        Student No: 23049710       |
-------------------------------------
```
