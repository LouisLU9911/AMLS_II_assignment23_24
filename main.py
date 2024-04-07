#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import logging
import os
import sys

from A.constants import (
    DEFAULT_BATCH_SIZE_PER_DEVICE,
    DEFAULT_DATASET_FOLDER,
    DEFAULT_DATASET_IMAGEFOLDER,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_EPOCHS,
    DEAULT_NUM_WORKERS_PER_DEVICE,
)
from A.logger import logger, set_log_level
from A.pretrained import PretrainedModel

CWD = os.getcwd()
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")


def setup_parse():
    import argparse

    description = "AMLS II Final Assignment"
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(whether_output=True)
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        "--seed",
        action="store",
        default=DEFAULT_RANDOM_SEED,
        metavar="S",
        type=int,
        nargs="?",
        help=f"random seed; default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--mode",
        action="store",
        default="train",
        help="train or inference",
    )
    parser.add_argument(
        "--batch",
        action="store",
        default=DEFAULT_BATCH_SIZE_PER_DEVICE,
        metavar="B",
        type=int,
        nargs="?",
        help=f"batch size per GPU; default: {DEFAULT_BATCH_SIZE_PER_DEVICE}",
    )
    parser.add_argument(
        "--epoch",
        action="store",
        default=DEFAULT_EPOCHS,
        metavar="E",
        type=int,
        nargs="?",
        help=f"number of train epochs; default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--workers",
        action="store",
        default=DEAULT_NUM_WORKERS_PER_DEVICE,
        help=f"number of workers per GPU; default: {DEAULT_NUM_WORKERS_PER_DEVICE}",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="push model to huggingface",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions provided")
    subparsers.required = True

    pretrained_parser = subparsers.add_parser("pretrained")
    pretrained_parser.add_argument(
        "-m",
        "--model_name",
        action="store",
        default=DEFAULT_PRETRAINED_MODEL,
        help=f"pretrained model name from huggingface; default: {DEFAULT_PRETRAINED_MODEL}",
    )
    pretrained_parser.add_argument(
        "--dataset",
        action="store",
        default=DEFAULT_DATASET_IMAGEFOLDER,
        help=f"path to dataset imagefolder; default: {DEFAULT_DATASET_IMAGEFOLDER}",
    )

    moe_parser = subparsers.add_parser("moe")
    moe_parser.add_argument(
        "-m",
        "--model_name",
        action="store",
        default=DEFAULT_PRETRAINED_MODEL,
        help=f"pretrained model name from timm; default: {DEFAULT_PRETRAINED_MODEL}",
    )
    moe_parser.add_argument(
        "--dataset",
        action="store",
        default=DEFAULT_DATASET_FOLDER,
        help=f"path to dataset imagefolder; default: {DEFAULT_DATASET_FOLDER}",
    )

    subparsers.add_parser("info")

    args, _ = parser.parse_known_args()
    return args


def print_info():
    print("-------------------------------------")
    print("|      AMLS II Assignment 23-24     |")
    print("|         Name: Zhaoyan Lu          |")
    print("|        Student No: 23049710       |")
    print("-------------------------------------")


def main():
    try:
        args = setup_parse()
        set_log_level(args.loglevel)

        if args.action == "info":
            print_info()
        elif args.action == "moe":
            # TODO
            pass
        elif args.action == "pretrained":
            num_workers = args.workers * len(CUDA_VISIBLE_DEVICES.split(","))
            model = PretrainedModel(
                CWD,
                args.model_name,
                args.dataset,
                args.batch,
                num_workers=num_workers,
                epoch=args.epoch,
                seed=args.seed,
            )
            if args.mode == "train":
                model.train(save_model=args.save, push_to_hub=args.push_to_hub)
            elif args.mode == "inference":
                # TODO
                model.inference()
            else:
                raise Exception(
                    f"Invalid solve mode: {args.mode}; Please set --mode MODE!"
                )
        else:
            raise Exception(f"Unsupported action: {args.action}")
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
