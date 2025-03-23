import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class ImageSegmentationDatasetDFC(Dataset):
    def __init__(self, root, set_name, processor, transform=None):
        self.root = root
        self.set_name = set_name
        self.processor = processor
        self.transform = transform
        self.image_path = os.path.join(self.root, self.set_name)
        self.coco = COCO(os.path.join(self.root, 'COCO_LABELS/updated2_coco.json'))
        self.ids = self.coco.getImgIds()
        self.ids.sort()

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.image_path, self.coco.loadImgs(img_id)[0]['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        class_map = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        instance_map = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        for idx, ann in enumerate(annotations):
            binary_mask = self.coco.annToMask(ann)
            class_id = ann['category_id']
            instance_id = idx + 1  # Start instance IDs from 1

            class_map[binary_mask > 0] = class_id
            instance_map[binary_mask > 0] = instance_id

        class_labels = np.unique(class_map)
        image = np.array(image)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_map)
            image, instance_map = transformed['image'], transformed['mask']
            image = image.transpose(2,0,1)

        inst2class = {inst: 1 for inst in np.unique(instance_map)}
        inst2class[0] = 0
        
        if class_labels.shape[0] == 1 and class_labels[0] == 0 or len(inst2class) == 1:
            # Some image does not have annotation (all ignored)
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k:v.squeeze() for k,v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            inputs = self.processor([image], [instance_map], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}
        return inputs

    def __len__(self):
        return len(self.ids)
