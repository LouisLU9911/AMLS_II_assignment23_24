#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess datasets"""

import os
import json
import shutil

import pandas as pd
from tqdm import tqdm

cwd = os.getcwd()

IMAGEFOLDER = "imagefolder"
TRAIN_IMAGES_DIRNAME = "train_images"
TRAIN_IMAGE_FOLDER = "train"


def cp_dataset():
    # create train ImageFolder
    train_image_folder_path = os.path.join(cwd, IMAGEFOLDER, TRAIN_IMAGE_FOLDER)
    if not os.path.exists(os.path.join(cwd, IMAGEFOLDER)):
        os.mkdir(os.path.join(cwd, IMAGEFOLDER))
        if not os.path.exists(train_image_folder_path):
            os.mkdir(train_image_folder_path)
            print(f"mkdir {train_image_folder_path} done!")

    label_num_to_disease_map = get_label_num_to_disease_map()
    for label_num in label_num_to_disease_map.keys():
        dirpath = os.path.join(train_image_folder_path, label_num)
        os.mkdir(dirpath)
        print(f"mkdir {dirpath} done!")

    train_df = pd.read_csv(os.path.join(cwd, "train.csv"))
    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        src_path = os.path.join(cwd, TRAIN_IMAGES_DIRNAME, row["image_id"])
        dst_path = os.path.join(
            train_image_folder_path, str(row["label"]), row["image_id"]
        )
        shutil.copy(src_path, dst_path)
    print(f"Copy from the original datasets to {train_image_folder_path} successfully!")


def get_label_num_to_disease_map() -> dict:
    with open(os.path.join(cwd, "label_num_to_disease_map.json"), "r") as f:
        label_num_to_disease_map = json.load(f)
    return label_num_to_disease_map


if __name__ == "__main__":
    cp_dataset()
