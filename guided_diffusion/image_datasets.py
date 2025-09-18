import json
from collections import defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def create_dataloader(
    image_dir,
    label_dir,
    conds_json,
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
        conds_json,
        image_suffix=image_suffix,
        label_suffix=label_suffix,
        transforms=transforms,
        num_classes=num_classes,
        grayscale=grayscale,
    )

    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=2,
        )
    while True:
        yield from loader


class SDMDataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_dir,
        conds_json: str | Path | None = None,
        image_suffix=".jpg",
        label_suffix=".png",
        transforms: A.Compose | None = None,
        num_classes: int = 19,  # 0(背景)を含んだクラス数
        grayscale: bool = False,
    ):
        self.grayscale = grayscale
        self.num_classes = num_classes
        # 画像のパスについて確認
        assert image_dir is not None
        assert label_dir is not None
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        assert self.image_dir.exists()
        assert self.label_dir.exists()
        # 条件のjsonがあればパース
        key_num = 0
        data: dict[str, dict[str, str | int]] = dict()
        if conds_json:
            with open(conds_json, "r") as f:
                data = json.load(f)
            key_num = len(data.keys())
        # 条件を満たしたデータを格納
        self.image_pathes: list[Path] = list()
        self.label_pathes: list[Path] = list()
        self.conds: dict[str, list] = defaultdict(list)
        files = sorted(self.image_dir.glob(f"*{image_suffix}"))
        for image_path in tqdm(files, desc="loading dataset"):
            name = image_path.stem
            # セマンティックラベル
            label_path = self.label_dir.joinpath(name + label_suffix)
            # 条件ラベル
            cond = dict()
            for key, mapping in data.items():
                if name in mapping:
                    cond[key] = mapping[name]
            # 条件を満たしたら採用
            if label_path.exists() and (len(cond.keys()) == key_num):
                self.image_pathes.append(image_path)
                self.label_pathes.append(label_path)
                for key, v in cond.items():
                    self.conds[key].append(v)
        print("total data num:", len(self.image_pathes))
        # データ拡張
        assert transforms is not None

    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, i: int):
        # 画像
        image: Image.Image = Image.open(self.image_pathes[i])
        if self.grayscale:
            img = np.array(image.convert("L"))[..., None]
        else:
            img = np.array(image.convert("RGB"))
        mask = np.array(Image.open(self.label_pathes[i]))  # (H, W) int labels
        mask = mask.clip(0, self.num_classes - 1)

        # Augmentation
        aug = self.transforms(image=img, mask=mask)
        img: np.ndarray = aug["image"]
        mask: np.ndarray = aug["mask"]

        # normalization
        img = img.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        # 条件
        conds = {k: self.conds[k][i] for k in self.conds}

        # 整形
        out_dict = dict()
        out_dict["path"] = self.image_pathes[i].as_posix()
        out_dict["label"] = np.expand_dims(aug["mask"], 0)
        if len(conds) > 0:
            out_dict["conds"] = conds
        return img, out_dict
