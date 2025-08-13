from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    image_dir,
    label_dir,
    image_size,
    num_classes,
    batch_size,
    num_workers,
    deterministic=False,
    transforms=None,
    image_suffix=".jpg",
    label_suffix=".png",
    grayscale=False,
):
    dataset = SDMDataset(
        image_dir,
        label_dir,
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        transforms=transforms,
        num_classes=num_classes,
        size=image_size,
        grayscale=grayscale,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
    while True:
        yield from loader


class SDMDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        image_suffix=".jpg",
        label_suffix=".png",
        transforms: A.Compose | None = None,
        num_classes: int = 19,
        size: int = 256,
        grayscale: bool = False,
    ):
        assert image_dir is not None
        assert label_dir is not None
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.grayscale = grayscale
        self.num_classes = num_classes
        assert self.image_dir.exists()
        assert self.label_dir.exists()
        self.image_pathes: list[Path] = list()
        self.label_pathes: list[Path] = list()
        for image_path in self.image_dir.glob(f"*{image_suffix}"):
            label_path = self.label_dir.joinpath(image_path.stem + label_suffix)
            if label_path.exists():
                self.image_pathes.append(image_path)
                self.label_pathes.append(label_path)
        if transforms is None:
            self.transforms = A.Compose(
                [
                    A.Resize(size, size, interpolation=Image.Resampling.BILINEAR),
                    A.HorizontalFlip(),
                ]
            )

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, i: int):
        image: Image.Image = Image.open(self.image_pathes[i])
        if self.grayscale:
            img = np.array(image.convert("L"))
        else:
            img = np.array(image.convert("RGB"))
        mask = np.array(Image.open(self.label_pathes[i]))  # (H, W) int labels

        # Augmentation
        aug = self.transforms(image=img, mask=mask)
        img: np.ndarray = aug["image"]
        mask: np.ndarray = aug["mask"]

        # normalization
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        out_dict = dict()
        out_dict["path"] = self.image_pathes[i].as_posix()
        out_dict["label"] = np.expand_dims(aug["mask"], 0)
        return img, out_dict
