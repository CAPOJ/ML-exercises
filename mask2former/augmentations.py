import numpy as np
import albumentations as albu
from albumentations.augmentations.geometric.rotate import RandomRotate90


def get_transforms(
        width: int,
        height: int,
        dataset_name: str,
        preprocessing: bool = True,
        augmentations: bool = True,
        postprocessing: bool = True,
) -> albu.BaseCompose:
    transforms = []

    if dataset_name == 'dfc':
        if preprocessing:
            transforms.append(albu.RandomCrop(width=width, height=height))
    else:
        if preprocessing:
            transforms.append(albu.Resize(width=width, height=height))

    if augmentations:
        transforms.extend([RandomRotate90(p=0.5)])

    if postprocessing:
        transforms.extend(
            [
                albu.Normalize(
                        mean=np.array([123.675, 116.280, 103.530]) / 255,
                        std=np.array([58.395, 57.120, 57.375]) / 255,
                    ),  
            ]
        )

    return albu.Compose(transforms)