from config import Config
from metrics import evaluate_map
from train_utils import load_object

import torch
from transformers import Mask2FormerForUniversalSegmentation
from torchmetrics.detection import MeanAveragePrecision
import pytorch_lightning as pl


class Mask2Former(pl.LightningModule):
    def __init__(self, config: Config, processor):
        super(Mask2Former, self).__init__()
        self._config = config
        self.processor = processor
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self._config.pretrained_model,
            id2label=self._config.id2label,
            label2id={v:k for k,v in self._config.id2label.items()},
            ignore_mismatched_sizes=True,
            )
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.model.pixel_level_module.decoder.parameters():
        #     param.requires_grad = True
        # for param in self.model.model.transformer_module.parameters():
        #     param.requires_grad = True
        # for param in self.model.class_predictor.parameters():
        #     param.requires_grad = True
        
        self.train_metric = MeanAveragePrecision(iou_type='segm')
        self.val_metric = MeanAveragePrecision(iou_type='segm')
        self.train_outputs, self.val_outputs = [], []
        #######################
        self.input_sample = None
        ########################
        self.save_hyperparameters()

    def forward(self, images, masks_labels, class_labels):
        outputs = self.model(
            pixel_values=images,
            mask_labels=masks_labels,
            class_labels=class_labels
        )

        return outputs

    def training_step(self, batch):
        images = batch['pixel_values']
        ###############################
        print(f"\n\tРАЗМЕР И ПАРАМЕТРЫ INPUT SAMPLE {images.shape} ТИП ДАННЫХ {images.dtype}\n")
        masks_labels = [labels for labels in batch["mask_labels"]]
        print(f"\n\tРАЗМЕР И ПАРАМЕТРЫ MASK LABELS:")
        print(f"  Количество элементов в masks_labels: {len(masks_labels)}")
        for i, item in enumerate(masks_labels):
            print(f"  Элемент {i}:")
            print(f"    Тип: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"    Размерность: {item.shape}")
                print(f"    Тип данных: {item.dtype}")
            elif isinstance(item, list):
                print(f"    Количество элементов: {len(item)}")
        class_labels = [labels for labels in batch["class_labels"]]
        print(f"\n\tРАЗМЕР И ПАРАМЕТРЫ CLASS LABELS:")
        print(f"  Количество элементов в class_labels: {len(class_labels)}")
        for i, item in enumerate(class_labels):
            print(f"  Элемент {i}:")
            print(f"    Тип: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"    Размерность: {item.shape}")
                print(f"    Тип данных: {item.dtype}")
            elif isinstance(item, list):
                print(f"    Количество элементов: {len(item)}")
        outputs = self.forward(images, masks_labels, class_labels)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            return_binary_maps=True,
            target_sizes=[(self._config.data_config.processor_image_size,
                           self._config.data_config.processor_image_size)] * len(images),
        )

        preds_map, targets_map = evaluate_map(batch, results)
        self.train_metric.update(preds_map, targets_map)

        loss = outputs.loss
        self.train_outputs.append(loss)

        return({'loss': loss})

    def validation_step(self, batch):
        images = batch['pixel_values']
        masks_labels = [labels for labels in batch["mask_labels"]]
        class_labels = [labels for labels in batch["class_labels"]]
        names = [labels for labels in batch["image_paths"]]
        outputs = self.forward(images, masks_labels, class_labels)
        

        results = self.processor.post_process_instance_segmentation(
            outputs,
            return_binary_maps=True,
            target_sizes=[(self._config.data_config.processor_image_size,
                          self._config.data_config.processor_image_size)] * len(images),
        )
        print("Results after post-processing val: ", results)
        loss = outputs.loss
        self.val_outputs.append(loss)
        preds_map, targets_map = evaluate_map(batch, results)
        #Проверка на несповпадение
        try:
            self.val_metric.update(preds_map, targets_map)
        except ValueError as e:
            print(f"Ошибка не совпадние маски и лэйбла: {str(e)} ")
            print(f'Имена батча {names}')
        except Exception as e:
            print(f"Новая вообще ошибка: {e}")

    def predict_step(self, batch):
        images = batch['pixel_values']
        masks_labels = [labels for labels in batch["mask_labels"]]
        class_labels = [labels for labels in batch["class_labels"]]

        outputs = self.forward(images, masks_labels, class_labels)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            return_binary_maps=True,
            target_sizes=[(self._config.data_config.processor_image_size,
                          self._config.data_config.processor_image_size)] * len(images),
        )

        return results

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_outputs).mean()

        for key, value in self.train_metric.compute().items():
            self.log(f"train_{key}", value, on_epoch=True)

        self.log('train_loss', avg_loss, on_epoch=True)

        self.train_outputs.clear()
        self.train_metric.reset()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_outputs).mean()

        for key, value in self.val_metric.compute().items():
            self.log(f"val_{key}", value, on_epoch=True)

        self.log('val_loss', avg_loss, on_epoch=True)
        
        self.val_outputs.clear()
        self.val_metric.reset()

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self.model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }