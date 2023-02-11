import io
import json
import logging
import os

import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop((1333, 800), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

def get_panoptic_transforms(stage: str):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.RandomHorizontalFlip(),
            trn.RandomCrop((1333, 800), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((1333, 800)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='/root/Projects/openpsg/ce7454/data',
        num_classes=56,
        is_fp16=False,
        two_stage=False,
    ):
        super(PSGClsDataset, self).__init__()
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.root = root
        self.transform_image = get_transforms(stage)
        self.transform_pimage = get_panoptic_transforms(stage)
        self.num_classes = num_classes
        self.is_fp16 = is_fp16
        self.two_stage = two_stage

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        img_path = os.path.join(self.root, 'coco', sample['file_name'])
        pimag_path = os.path.join(self.root, 'detr2', sample['file_name'])
        try:
            with open(img_path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                img = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(img_path))
            raise e
        try:
            with open(pimag_path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                pimage = Image.open(buff).convert('RGB')
                pimg = self.transform_pimage(pimage)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(pimag_path))
            raise e
        
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        
        if self.two_stage:
            return pimg, soft_label
        else:
            return img, soft_label
