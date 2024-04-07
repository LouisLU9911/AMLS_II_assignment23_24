import os
from typing import List


import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LeafDiseaseDataset(Dataset):

    def __init__(
        self,
        img_labels: pd.DataFrame,
        img_dir: str,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def build_dataset(
    annotations_file: str,
    img_dir: str,
    seed: int = 42,
    test_size: float = 0.1,
    allowed_labels: List[str] = None,
    train_transform=None,
    test_transform=None,
    train_target_transform=None,
    test_target_transform=None,
):
    """Build train and test Datasets for training."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(annotations_file)

    # Filter allowed labels for different experts
    if allowed_labels:
        conditions = df["label"] == -1
        for label in allowed_labels:
            conditions |= df["label"] == label
        df = df[conditions]

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )

    train_ds = LeafDiseaseDataset(
        train_df,
        img_dir=img_dir,
        transform=train_transform,
        target_transform=train_target_transform,
    )
    test_ds = LeafDiseaseDataset(
        test_df, img_dir=img_dir, transform=test_transform, target_transform=test_target_transform
    )
    return train_ds, test_ds


def build_transform(image_processor=None, pretrained_cfg=None):
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
        size = (pretrained_cfg['input_size'][1], pretrained_cfg['input_size'][2])
        crop_size = size
        normalize = Normalize(
            mean=pretrained_cfg["mean"], std=pretrained_cfg["std"]
        )
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
