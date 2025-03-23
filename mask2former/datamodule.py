from typing import Optional
from config import DataConfig
from dataset import ImageSegmentationDataset
from dataset_dfc import ImageSegmentationDatasetDFC
from augmentations import get_transforms
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig, processor):
        super().__init__()
        self._config = config
        self.processor = processor
        self.batch_size = self._config.batch_size
        self.n_workers = self._config.n_workers

    def setup(self, stage: Optional[str] = None):
        
        if self._config.dataset_name == 'dfc': 
            self.train_dataset = ImageSegmentationDatasetDFC(
                self._config.data_path,
                'RGB',
                processor=self.processor,
                transform=get_transforms(self._config.processor_image_size,
                                        self._config.processor_image_size,
                                        self._config.dataset_name,
                                        )
            )
            self.val_dataset = ImageSegmentationDatasetDFC(
                self._config.data_path,
                'RGB',
                processor=self.processor,
                transform=get_transforms(self._config.processor_image_size,
                                        self._config.processor_image_size,
                                        self._config.dataset_name,
                                        )
            )
        else:
            self.train_dataset = ImageSegmentationDataset(
                self._config.data_path,
                'train',
                processor=self.processor,
                transform=get_transforms(self._config.processor_image_size,
                                        self._config.processor_image_size,
                                        self._config.dataset_name,
                                        )
            )
            self.val_dataset = ImageSegmentationDataset(
                self._config.data_path,
                'valid',
                processor=self.processor,
                transform=get_transforms(self._config.processor_image_size,
                                        self._config.processor_image_size,
                                        self._config.dataset_name,
                                        )
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        pixel_values = torch.stack([example["pixel_values"] for example in batch])
        pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
        class_labels = [example["class_labels"] for example in batch]
        mask_labels = [example["mask_labels"] for example in batch]
        images_paths = [example['image_paths'] for example in batch]
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "class_labels": class_labels,
            "mask_labels": mask_labels,
            "image_paths" : images_paths,
        }
        # pixel_values = torch.stack([example[0]["pixel_values"] for example in batch])  # Получаем inputs
        # pixel_mask = torch.stack([example[0]["pixel_mask"] for example in batch])
        # class_labels = [example[0]["class_labels"] for example in batch]
        # mask_labels = [example[0]["mask_labels"] for example in batch]
        # image_paths = [example[1] for example in batch]  # Получаем названия изображений
        # return {
        #     "pixel_values": pixel_values,
        #     "pixel_mask": pixel_mask,
        #     "class_labels": class_labels,
        #     "mask_labels": mask_labels,
        #     "image_paths": image_paths,  # Добавляем названия изображений
        # }
