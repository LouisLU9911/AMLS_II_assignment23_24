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
This project aims at solving the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/) task on [Kaggle](https://www.kaggle.com/) using Deep Neural Network (DNN).

## Repo Structure

## Requirements

* Miniconda / Anaconda
* Nvidia GPU with cuda 12.2

```
# download the dataset from kaggle
kaggle
# format python code
black[jupyter]
# data preprocess
pandas
tqdm
```

Create conda env:

```bash
$ make create-env
# or
$ conda env create -f environment.yml
```

## Datasets

We are using the dataset provided by the [competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data).

```bash
# Step 0: enter conda env
$ conda activate amls_ii-final-uceezl8
# Step 1: download the dataset and extract it
$ make datasets
# or
$ cd Datasets
$ kaggle competitions download -c cassava-leaf-disease-classification
$ unzip -q cassava-leaf-disease-classification.zip
$ python prepare.py
```

## Usage

