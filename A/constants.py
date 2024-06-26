#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Constants"""

DEFAULT_PRETRAINED_MODEL = "facebook/convnextv2-base-1k-224"

DEFAULT_RANDOM_SEED = 42

DEFAULT_BATCH_SIZE_PER_DEVICE = 70
DEAULT_NUM_WORKERS_PER_DEVICE = 5

DEFAULT_EPOCHS = 16

# For models from transforms
DATADIR = "Datasets"
# default imagefolder
DEFAULT_DATASET_IMAGEFOLDER = DATADIR + "/imagefolder"
DATALOADER_PREFETCH_FACTOR = 1

DEFAULT_CONFIG_PATH = "A/config.json"

# For MoE
# default dataset folder
DEFAULT_DATASET_FOLDER = DATADIR + "/train_images"

DEFAULT_EXPERT_LABEL_MAPPING = {"BaseModel": [0, 1, 2, 3, 4]}

DEFAULT_HUGGINGFACE_ACCOUNT = "louislu9911"
