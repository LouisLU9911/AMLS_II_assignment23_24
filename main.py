#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import logging
import os
import sys

from A.constants import (
    DEFAULT_BATCH_SIZE_PER_DEVICE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_FOLDER,
    DEFAULT_DATASET_IMAGEFOLDER,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_EPOCHS,
    DEAULT_NUM_WORKERS_PER_DEVICE,
)
from A.logger import logger, set_log_level


CWD = os.getcwd()
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "")


def setup_parse():
    import argparse

    description = "AMLS II Final Assignment"
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(whether_output=True)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parent_parser.add_argument(
        "--seed",
        action="store",
        default=DEFAULT_RANDOM_SEED,
        metavar="S",
        type=int,
        nargs="?",
        help=f"random seed; default: {DEFAULT_RANDOM_SEED}",
    )
    parent_parser.add_argument(
        "--mode",
        action="store",
        default="train",
        help="train or inference",
    )
    parent_parser.add_argument(
        "--batch",
        action="store",
        default=DEFAULT_BATCH_SIZE_PER_DEVICE,
        metavar="B",
        type=int,
        nargs="?",
        help=f"batch size per GPU; default: {DEFAULT_BATCH_SIZE_PER_DEVICE}",
    )
    parent_parser.add_argument(
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
    parent_parser.add_argument(
        "--save",
        action="store_true",
        help="save model",
    )
    parent_parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="push model to huggingface",
    )
    parent_parser.add_argument(
        "--config",
        action="store",
        default=DEFAULT_CONFIG_PATH,
        help=f"path to config.json; default: {DEFAULT_CONFIG_PATH}",
    )

    subparsers = parser.add_subparsers(dest="action", help="actions provided")
    subparsers.required = True

    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
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

    moe_parser = subparsers.add_parser("moe", parents=[parent_parser])
    moe_parser.add_argument(
        "--model_name",
        action="store",
        default=DEFAULT_PRETRAINED_MODEL,
        help=f"pretrained model name from timm; default: {DEFAULT_PRETRAINED_MODEL}",
    )
    moe_parser.add_argument(
        "--dataset",
        action="store",
        default=DEFAULT_DATASET_FOLDER,
        help=f"path to images; default: {DEFAULT_DATASET_FOLDER}",
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
            from A.moe import train

            if args.mode == "train":
                train(
                    batch_size_per_device=args.batch,
                    num_workers_per_device=args.workers,
                    epoch=args.epoch,
                    seed=args.seed,
                )
            elif args.mode == "inference":
                # TODO
                model.inference()
            else:
                raise Exception(
                    f"Invalid solve mode: {args.mode}; Please set --mode MODE!"
                )
            pass
        elif args.action == "pretrained":
            from A.pretrained import PretrainedModel

            num_workers = args.workers * len(CUDA_VISIBLE_DEVICES.split(","))
            model = PretrainedModel(
                cwd=CWD,
                model_name=args.model_name,
                dataset_path=args.dataset,
                config=args.config,
                seed=args.seed,
            )
            if args.mode == "train":
                model.train(
                    epoch=args.epoch,
                    batch_size_per_device=args.batch,
                    num_workers=num_workers,
                    save_model=args.save,
                    push_to_hub=args.push_to_hub,
                )
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
