from lightning_module_ddp import Mask2Former
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import Mask2FormerImageProcessor
from config import Config


trained_model = Mask2Former.load_from_checkpoint('/files/private_data/maskformer2/modeling-mask2former/lightning_logs/version_245/checkpoints/epoch=0-step=8.ckpt')
onnx_path = '/files/shared_data/models/gb/ONNX'
batch_size = 8
channels = 3
height = 512
width = 512
input_sample = torch.zeros(batch_size, channels, height, width, dtype=torch.float32)
mask_labels = [torch.zeros(10, height, width, dtype=torch.float32) for _ in range(batch_size)]
class_labels = [torch.zeros(10, dtype=torch.int64) for _ in range(batch_size)]
print(input_sample.shape)  # Например: torch.Size([1, 3, 224, 224])
print(input_sample.dtype)  # Например: torch.float32
trained_model.to_onnx(onnx_path,[input_sample,mask_labels,class_labels],export_params=True,)