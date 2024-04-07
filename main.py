#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entrypoint of this assignment."""

import logging
import os
import sys

from utils.constants import (
    DATALOADER_PREFETCH_FACTOR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_FOLDER,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TRAIN_EPOCHS,
    DEAULT_NUM_WORKERS,
)
from utils.logger import logger, set_log_level


CWD = os.getcwd()


def setup_parse():
    import argparse

    description = "AMLS II Final Assignment"
    parser = argparse.ArgumentParser(description=description)
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
    parser.set_defaults(whether_output=True)

    subparsers = parser.add_subparsers(dest="action", help="actions provided")
    subparsers.required = True

    solve_subparser = subparsers.add_parser("solve")
    solve_subparser.add_argument(
        "-m",
        "--model_name",
        action="store",
        default=DEFAULT_PRETRAINED_MODEL,
        help=f"pretrained model name; default: {DEFAULT_PRETRAINED_MODEL}",
    )
    solve_subparser.add_argument(
        "--mode",
        action="store",
        help="training or inference",
    )
    solve_subparser.add_argument(
        "--dataset",
        action="store",
        default=DEFAULT_DATASET_FOLDER,
        help=f"path to dataset image folder; default: {DEFAULT_DATASET_FOLDER}",
    )
    solve_subparser.add_argument(
        "--seed",
        action="store",
        default=DEFAULT_RANDOM_SEED,
        metavar="S",
        type=int,
        nargs="?",
        help=f"random seed; default: {DEFAULT_RANDOM_SEED}",
    )
    solve_subparser.add_argument(
        "--batch",
        action="store",
        default=DEFAULT_BATCH_SIZE,
        metavar="B",
        type=int,
        nargs="?",
        help=f"batch size per GPU; default: {DEFAULT_BATCH_SIZE}",
    )
    solve_subparser.add_argument(
        "--n_epochs",
        action="store",
        default=DEFAULT_TRAIN_EPOCHS,
        metavar="E",
        type=int,
        nargs="?",
        help=f"number of training epochs; default: {DEFAULT_TRAIN_EPOCHS}",
    )
    solve_subparser.add_argument(
        "--n_workers",
        action="store",
        default=DEAULT_NUM_WORKERS,
        help=f"number of workers per GPU; default: {DEAULT_NUM_WORKERS}",
    )
    solve_subparser.add_argument(
        "--save",
        action="store_true",
        help="save model",
    )
    solve_subparser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="push model to huggingface",
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


def get_dataset(dataset_path: str):
    from datasets import load_dataset, load_metric

    train_image_folder = os.path.join(CWD, dataset_path)
    logger.info(f"Begin loading {train_image_folder} ...")
    dataset = load_dataset("imagefolder", data_dir=train_image_folder)
    logger.info(f"Load {train_image_folder} successfully!")
    return dataset


def get_preprocess_func(image_processor):
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

    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)

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


def training(
    dataset_path: str = DEFAULT_DATASET_FOLDER,
    model_name: str = DEFAULT_PRETRAINED_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_epochs: int = DEFAULT_TRAIN_EPOCHS,
    n_workers: int = DEAULT_NUM_WORKERS,
    seed: int = DEFAULT_RANDOM_SEED,
    save_model: bool = False,
    push_to_hub: bool = False,
):
    import numpy as np
    import torch
    from datasets import load_metric
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        TrainingArguments,
        Trainer,
    )

    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    NUM_WORKERS = n_workers * len(CUDA_VISIBLE_DEVICES.split(","))

    dataset = get_dataset(dataset_path)

    model_checkpoint = model_name

    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    preprocess_train, preprocess_val = get_preprocess_func(image_processor)

    splits = dataset["train"].train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        seed=seed,
    )
    train_ds = splits["train"]
    val_ds = splits["test"]
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    # set id 2 label for 5 different classes
    id2label = {i: i for i in range(5)}
    label2id = {i: i for i in range(5)}

    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    model_name = model_checkpoint.split("/")[-1]

    if save_model:
        save_strategy = "epoch"
    else:
        save_strategy = "no"

    args = TrainingArguments(
        f"{model_name}-finetuned-cassava-leaf-disease",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy=save_strategy,
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        seed=seed,
        metric_for_best_model="accuracy",
        push_to_hub=push_to_hub,
    )

    metric = load_metric("accuracy")

    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are the ground-truth labels as Numpy arrays.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    logger.info("Begin training...")
    train_results = trainer.train()
    logger.info("training ends")
    # rest is optional but nice to have
    if save_model:
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()


def inference():
    pass


def main():
    try:
        args = setup_parse()
        set_log_level(args.loglevel)

        if args.action == "info":
            print_info()
        elif args.action == "solve":
            if args.mode == "training":
                training(
                    dataset_path=args.dataset,
                    model_name=args.model_name,
                    batch_size=args.batch,
                    n_epochs=args.n_epochs,
                    n_workers=args.n_workers,
                    seed=args.seed,
                    save_model=args.save,
                    push_to_hub=args.push_to_hub,
                )
            elif args.mode == "inference":
                inference()
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
