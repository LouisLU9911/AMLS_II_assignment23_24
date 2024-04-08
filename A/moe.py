import os
from typing import List

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
)

from .dataset import build_dataset, build_transform
from .constants import (
    DEFAULT_NUM_CLASSES,
    DEFAULT_EPOCHS,
    DEFAULT_RANDOM_SEED,
    DEAULT_NUM_WORKERS_PER_DEVICE,
    DEFAULT_BATCH_SIZE_PER_DEVICE,
    DATALOADER_PREFETCH_FACTOR,
    DEFAULT_TIMM_MODEL,
    DEFAULT_DATASET_FOLDER,
    DEFAULT_ANNOTATIONS_FILE,
)
from .logger import logger


def train(
    model_name: str = DEFAULT_TIMM_MODEL,
    annotations_file: str = DEFAULT_ANNOTATIONS_FILE,
    batch_size_per_device: int = DEFAULT_BATCH_SIZE_PER_DEVICE,
    num_workers_per_device: int = DEAULT_NUM_WORKERS_PER_DEVICE,
    epoch: int = DEFAULT_EPOCHS,
    img_dir: str = DEFAULT_DATASET_FOLDER,
    seed: int = DEFAULT_RANDOM_SEED,
    save: bool = False,
    push_to_hub: bool = False,
):
    from datasets import load_metric

    model = timm.create_model(
        model_name, pretrained=True, num_classes=DEFAULT_NUM_CLASSES
    )
    pretrained_cfg = model.pretrained_cfg
    train_transform, test_transform = build_transform(pretrained_cfg=pretrained_cfg)
    train_ds, test_ds = build_dataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
        seed=seed,
        test_size=0.1,
        allowed_labels=None,
        train_transform=train_transform,
        test_transform=test_transform,
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    n_devices = torch.cuda.device_count()
    num_workers = num_workers_per_device * n_devices
    args = TrainingArguments(
        f"{model_name}-moe-base-model-leaf",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size_per_device,
        gradient_accumulation_steps=4,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=DATALOADER_PREFETCH_FACTOR,
        per_device_eval_batch_size=batch_size_per_device,
        num_train_epochs=epoch,
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
        eval_dataset=test_ds,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    # base_model = BaseModel(
    #     model=model,
    #     train_ds=train_ds,
    #     test_ds=test_ds,
    #     batch_size_per_device=batch_size_per_device,
    #     num_workers_per_device=num_workers_per_device,
    # )

    # base_model.train(epoch=epoch, save=save, push_to_hub=push_to_hub)


class BaseModel:
    def __init__(
        self,
        model,
        train_ds: Dataset = None,
        test_ds: Dataset = None,
        batch_size_per_device: int = DEFAULT_BATCH_SIZE_PER_DEVICE,
        num_workers_per_device: int = DEAULT_NUM_WORKERS_PER_DEVICE,
    ) -> None:
        self.model = model
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            logger.info(f"Use {n_devices} GPUs!")
            self.model = nn.DataParallel(self.model)

        self.batch_size_per_device = batch_size_per_device
        self.batch_size = self.batch_size_per_device * n_devices
        self.num_workers_per_device = num_workers_per_device
        self.num_workers = self.num_workers_per_device * n_devices

        self.train_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
        self.test_dataloader = DataLoader(
            test_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def _train_loop(self, dataloader: DataLoader, loss_fn, optimizer):
        from tqdm import tqdm

        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()
        with tqdm(total=len(dataloader.dataset)) as pbar:
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction and loss
                pred = self.model(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.update(self.batch_size)
                # loss, current = loss.item(), batch * batch_size + len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test_loop(self, dataloader: DataLoader, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        logger.info(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    def train(
        self,
        epoch: int = DEFAULT_EPOCHS,
        learning_rate: int = None,
        loss_fn=None,
        optimizer=None,
        save: bool = False,
        push_to_hub: bool = False,
    ):
        if not learning_rate:
            learning_rate = 1e-3
        if not loss_fn:
            # Initialize the loss function
            loss_fn = nn.CrossEntropyLoss()
        if not optimizer:
            # TODO: optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.to(self.device)
        for t in range(epoch):
            logger.info(f"Epoch {t+1}\n-------------------------------")
            self._train_loop(self.train_dataloader, loss_fn, optimizer)
            self._test_loop(self.test_dataloader, loss_fn)
        # TODO: save and push


# class ExpertModel:
#     def __init__(
#         self,
#         model_name: str = DEFAULT_TIMM_MODEL,
#         annotations_file: str = DEFAULT_ANNOTATIONS_FILE,
#         dataset_folder: str = DEFAULT_DATASET_FOLDER,
#         allowed_labels: List[int] = None,
#     ) -> None:
#         self.model = timm.create_model(
#             model_name, pretrained=True, num_classes=NUM_FINETUNE_CLASSES
#         )
#         pretrained_cfg = self.model.pretrained_cfg
#         self.device = (
#             "cuda"
#             if torch.cuda.is_available()
#             else "mps" if torch.backends.mps.is_available() else "cpu"
#         )
#         if torch.cuda.device_count() > 1:
#             logger.info(f"Use {torch.cuda.device_count()} GPUs!")
#             self.model = nn.DataParallel(self.model)

#         train_transform, test_transform = build_transform(pretrained_cfg=pretrained_cfg)
#         train_ds, test_ds = build_dataset(
#             annotations_file,
#             dataset_folder,
#             allowed_labels=allowed_labels,
#             train_transform=train_transform,
#             test_transform=test_transform,
#         )
#         self.train_dataloader = DataLoader(
#             train_ds, batch_size=batch_size, num_workers=NUM_WORKERS
#         )
#         self.test_dataloader = DataLoader(
#             test_ds, batch_size=batch_size, num_workers=NUM_WORKERS
#         )

#     def _train_loop(self, dataloader: DataLoader, loss_fn, optimizer):
#         from tqdm import tqdm

#         size = len(dataloader.dataset)
#         # Set the model to training mode - important for batch normalization and dropout layers
#         # Unnecessary in this situation but added for best practices
#         self.model.train()
#         with tqdm(total=len(dataloader.dataset)) as pbar:
#             for batch, (X, y) in enumerate(dataloader):
#                 X, y = X.to(self.device), y.to(self.device)
#                 # Compute prediction and loss
#                 pred = self.model(X)
#                 loss = loss_fn(pred, y)

#                 # Backpropagation
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 pbar.update(batch_size)
#                 # loss, current = loss.item(), batch * batch_size + len(X)
#                 # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#     def _test_loop(self, dataloader: DataLoader, loss_fn):
#         # Set the model to evaluation mode - important for batch normalization and dropout layers
#         # Unnecessary in this situation but added for best practices
#         self.model.eval()
#         size = len(dataloader.dataset)
#         num_batches = len(dataloader)
#         test_loss, correct = 0, 0

#         # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#         # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#         with torch.no_grad():
#             for X, y in dataloader:
#                 X, y = X.to(self.device), y.to(self.device)
#                 pred = self.model(X)
#                 test_loss += loss_fn(pred, y).item()
#                 correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#         test_loss /= num_batches
#         correct /= size
#         logger.info(
#             f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
#         )

#     def train(
#         self,
#         learning_rate=None,
#         loss_fn=None,
#         optimizer=None,
#         save: bool = False,
#         push_to_hub: bool = False,
#     ):
#         if not learning_rate:
#             learning_rate = 1e-3
#         if not loss_fn:
#             # Initialize the loss function
#             loss_fn = nn.CrossEntropyLoss()
#         if not optimizer:
#             # TODO: optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

#         self.model.to(self.device)
#         for t in range(epochs):
#             logger.info(f"Epoch {t+1}\n-------------------------------")
#             self._train_loop(self.train_dataloader, loss_fn, optimizer)
#             self._test_loop(self.test_dataloader, loss_fn)
#         # TODO: save and push


# class MoEModel:
#     def __init__(
#         self,
#     ) -> None:
#         # TODO
#         pass

#     def train(self, expert_label_mapping: dict):
#         # TODO
#         pass

#     def inference(self):
#         pass
