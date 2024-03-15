# AMLS_II_assignment23_24

This repo is for the final assignment of ELEC0135 Applied Machine Learning Systems II (23/24).

## Contents

- [Overview](#Overview)
- [Repo Structure](#Repo-Structure)
- [Requirements](#Requirements)
- [Datasets](#Datasets)
- [Usage](#Usage)

## Overview

This project aims at solving the [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/) task on [Kaggle](https://www.kaggle.com/) using Deep Neural Network (DNN).

## Repo Structure

## Requirements

* Miniconda / Anaconda
* Nvidia GPU with cuda 12.2

```
# download the dataset from kaggle
kaggle==1.6.6
# format python code
black==24.2.0
```

Create conda env:

```bash
$ make create-env
# or
$ conda env create -f environment.yml
```

## Datasets

Download the dataset:

```bash
# Step 0: enter conda env
$ conda activate amls_ii-final-uceezl8
# Step 1: download the dataset and extract it
$ make datasets
# or
$ cd Datasets
$ kaggle competitions download -c cassava-leaf-disease-classification
$ unzip -q cassava-leaf-disease-classification.zip
```

## Usage

