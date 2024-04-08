import os
import json
from typing import List

from .constants import (
    DATALOADER_PREFETCH_FACTOR,
    DEFAULT_BATCH_SIZE_PER_DEVICE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_IMAGEFOLDER,
    DEFAULT_EPOCHS,
    DEFAULT_EXPERT_LABEL_MAPPING,
    DEFAULT_PRETRAINED_MODEL,
    DEFAULT_RANDOM_SEED,
)
from .logger import logger
from .dataset import get_preprocess_func


class PretrainedModel:
    def __init__(
        self,
        cwd: str,
        model_name: str = DEFAULT_PRETRAINED_MODEL,
        dataset_path: str = DEFAULT_DATASET_IMAGEFOLDER,
        config: str = DEFAULT_CONFIG_PATH,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        logger.info(f"Using {model_name}")
        self.cwd = cwd
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.seed = seed
        with open(config, "r") as f:
            config = json.load(f)
        self.expert_label_mapping = config.get(
            "expert_label_mapping", DEFAULT_EXPERT_LABEL_MAPPING
        )

    def _get_dataset(self):
        from datasets import load_dataset

        train_image_folder = os.path.join(self.cwd, self.dataset_path)
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
            seed=self.seed,
        )
        train_ds = splits["train"]
        val_ds = splits["test"]
        return train_ds, val_ds

    def _filter_labels(self, ds, labels: List[str]):
        filtered_ds = ds.filter(
            function=lambda x: x in labels,
            input_columns=["label"],
            num_proc=10,
        )
        return filtered_ds

    def train(
        self,
        epoch: int = DEFAULT_EPOCHS,
        batch_size_per_device: int = DEFAULT_BATCH_SIZE_PER_DEVICE,
        num_workers: int = 1,
        save_model: bool = False,
        push_to_hub: bool = False,
    ):
        import numpy as np
        import torch
        import evaluate
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            TrainingArguments,
            Trainer,
        )

        train_ds, val_ds = self._get_dataset()

        model_checkpoint = self.model_name
        image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        preprocess_train, preprocess_val = get_preprocess_func(image_processor)

        # set id 2 label for 5 different classes
        id2label = {i: i for i in range(5)}
        label2id = {i: i for i in range(5)}

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

        # Train all experts
        for expert_name, expert_labels in self.expert_label_mapping.items():
            labels_str = "_".join(map(str, expert_labels))
            expert_model_name = f"{expert_name}-leaf-disease-{labels_str}"

            # Filter labels for each expert
            expert_train_ds = self._filter_labels(train_ds, expert_labels)
            expert_val_ds = self._filter_labels(val_ds, expert_labels)
            logger.info(expert_train_ds)

            expert_train_ds.set_transform(preprocess_train)
            expert_val_ds.set_transform(preprocess_val)

            args = TrainingArguments(
                expert_model_name,
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
                train_dataset=expert_train_ds,
                eval_dataset=expert_val_ds,
                tokenizer=image_processor,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
            )
            logger.info(f"Begin training for {expert_model_name}...")
            train_results = trainer.train()
            logger.info(f"Training for {expert_model_name} ends")
            # rest is optional but nice to have
            if save_model:
                trainer.save_model()
                trainer.log_metrics("train", train_results.metrics)
                trainer.save_metrics("train", train_results.metrics)
                trainer.save_state()

    def inference(self):
        pass
