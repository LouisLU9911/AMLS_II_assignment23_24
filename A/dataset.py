#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset"""

import os
from .logger import logger
from .constants import DEFAULT_RANDOM_SEED


def get_dataset(
    cwd: str,
    dataset_path: str,
    seed: int = DEFAULT_RANDOM_SEED,
):
    from datasets import load_dataset

    train_image_folder = os.path.join(cwd, dataset_path)
    logger.info(f"Begin loading {train_image_folder} ...")
    dataset = load_dataset(
        "imagefolder",
        data_dir=train_image_folder,
        keep_in_memory=True,
        trust_remote_code=True,
    )
    logger.info(f"Load {train_image_folder} successfully!")
    splits = dataset["train"].train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        seed=seed,
    )
    train_ds = splits["train"]
    val_ds = splits["test"]
    return train_ds, val_ds


def build_transform(image_processor=None, pretrained_cfg=None):
    """Return transform for train set and test set."""
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        Normalize,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        RandomResizedCrop,
        Resize,
        ToTensor,
    )

    if image_processor:
        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )
        if "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"])
            crop_size = size
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
            crop_size = (size, size)
    elif pretrained_cfg:
        size = (pretrained_cfg["input_size"][1], pretrained_cfg["input_size"][2])
        crop_size = size
        normalize = Normalize(mean=pretrained_cfg["mean"], std=pretrained_cfg["std"])
    else:
        raise Exception("Need to provide image_processor or pretrained_cfg!")

    train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )
    return train_transforms, val_transforms


def get_preprocess_func(image_processor):
    """Return preprocess functions for AutoModelForImageClassification class"""
    train_transforms, val_transforms = build_transform(image_processor)

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    return preprocess_train, preprocess_val
