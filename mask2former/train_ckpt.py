import os
import argparse
from clearml import Task
from clearml_log import clearml_logging

from config import Config
from datamodule import DataModule

from lightning_module_CAWR import Mask2Former
from predict_callback import PredictAfterValidationCallback
from get_plot_for_images import *
# import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import Mask2FormerImageProcessor

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()

def train(config: Config, config_file):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
    # ddp = DDPStrategy(process_group_backend="gloo",timeout=datetime.timedelta(seconds=100000))
    task = Task.init(project_name=config.project_name, task_name=config.task)
    logger = task.get_logger()
    clearml_logging(config, logger)

    processor = Mask2FormerImageProcessor(
        reduce_labels=True,
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    checkpoint_callback = ModelCheckpoint(
        filename='mask2former-{epoch}-{step}-{val_loss:.4f}-{val_map:.4f}',
        save_top_k=1,
        every_n_epochs=1,
        verbose=True,
        mode='max',
        monitor=config.monitor_metric,
    )


    early_stopping_callback = EarlyStopping(monitor='val_map', patience=10)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    datamodule = DataModule(config.data_config, processor)

    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(
        max_epochs=config.n_epochs, 
        accelerator=config.accelerator, 
        devices=1, 
        log_every_n_steps=20,
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
            PredictAfterValidationCallback(logger=logger, mapping_file='datasetimages.json'),
            lr_monitor,
        ],
        # strategy=ddp,
        profiler='advanced',
    )
    model = Mask2Former(config, processor)

    task.upload_artifact(
        name='config_file',
        artifact_object=config_file,
    )
    trained_model = Mask2Former.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.fit(model=trained_model,
                ckpt_path=config.ckpt_path,
                datamodule=datamodule,
    )
    trained_model.model.save_pretrained('saved_models/mask2former_for_arseniy/')
    task.upload_artifact(
        name='best_transformers_model',
        artifact_object='saved_models/mask2former_for_arseniy'
    )
    names_of_sets,len_of_sets = show_sets('datasetimages.json',config.data_config.data_path)
    logger.report_histogram(
        title='Data distribution',
        series="Data Info",
        values=len_of_sets,
        xlabels=names_of_sets,
        xaxis="Названия датасэтов",
        yaxis="Кол-во снимков",
    )


if __name__ == "__main__":
    args = arg_parse()
    pl.seed_everything(42, workers=True)
    config = Config.from_yaml(args.config_file)
    train(config, args.config_file)
