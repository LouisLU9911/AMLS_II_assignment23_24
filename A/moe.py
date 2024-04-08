import os
from typing import List

from .dataset import get_preprocess_func, get_dataset
from .constants import (
    DEFAULT_EPOCHS,
    DEFAULT_DATASET_IMAGEFOLDER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_BATCH_SIZE_PER_DEVICE,
    DATALOADER_PREFETCH_FACTOR,
    DEFAULT_PRETRAINED_MODEL,
)
from .logger import logger

import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoImageProcessor,
    AutoModelForImageClassification,
)


class SwitchGate:
    def __init__(
        self,
        cwd: str,
        model_name: str = DEFAULT_PRETRAINED_MODEL,
        dataset_path: str = DEFAULT_DATASET_IMAGEFOLDER,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.cwd = cwd
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.seed = seed

    def _map_labels(self, ds, label_mapping: dict):
        def map_label(example):
            example["label"] = label_mapping[example["label"]]
            return example

        result_ds = ds.map(
            function=map_label,
            # input_columns=["label"],
            num_proc=10,
        )
        return result_ds

    def train(
        self,
        epoch: int = DEFAULT_EPOCHS,
        batch_size_per_device: int = DEFAULT_BATCH_SIZE_PER_DEVICE,
        num_workers: int = 1,
        save_model: bool = False,
        push_to_hub: bool = False,
    ):
        import evaluate

        train_ds, val_ds = get_dataset(
            cwd=self.cwd, dataset_path=self.dataset_path, seed=self.seed
        )

        label_mapping = {0: 0, 4: 0, 1: 1, 2: 1, 3: 1}
        train_ds = self._map_labels(train_ds, label_mapping)
        val_ds = self._map_labels(val_ds, label_mapping)

        model_checkpoint = self.model_name
        image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        preprocess_train, preprocess_val = get_preprocess_func(image_processor)

        # set id 2 label for 2 experts
        id2label = {i: i for i in range(2)}
        label2id = {i: i for i in range(2)}

        model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )
        model_checkpoint = model_checkpoint.split("/")[-1]

        if save_model:
            save_strategy = "epoch"
            load_best_model_at_end = True
        else:
            save_strategy = "no"
            load_best_model_at_end = False

        switch_gate_name = f"switch_gate-leaf-disease-{model_checkpoint}"

        train_ds.set_transform(preprocess_train)
        val_ds.set_transform(preprocess_val)

        args = TrainingArguments(
            switch_gate_name,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy=save_strategy,
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size_per_device,
            gradient_accumulation_steps=4,
            dataloader_num_workers=num_workers,
            dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
            per_device_eval_batch_size=batch_size_per_device,
            num_train_epochs=epoch,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=load_best_model_at_end,
            seed=self.seed,
            metric_for_best_model="accuracy",
            push_to_hub=push_to_hub,
        )

        metric = evaluate.load("accuracy")

        # the compute_metrics function takes a Named Tuple as input:
        # predictions, which are the logits of the model as Numpy arrays,
        # and label_ids, which are the ground-truth labels as Numpy arrays.
        def compute_metrics(eval_pred):
            """Computes accuracy on a batch of predictions"""
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(
                predictions=predictions, references=eval_pred.label_ids
            )

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
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
        logger.info(f"Begin training for {switch_gate_name}...")
        train_results = trainer.train()
        logger.info(f"Training for {switch_gate_name} ends")
        # rest is optional but nice to have
        if save_model:
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

    def test(self):
        pass
