
import torch
from detectron2.utils.logger import setup_logger

setup_logger()
import argparse
import json

import cv2

# import some common libraries
import numpy as np
import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
import torch
import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import torchvision.transforms as T
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(
#     model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# )
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well

# model = build_model(cfg)
# DetectionCheckpointer(model).load(
#     "https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl"
# )
# model.eval()
# with torch.cuda.device(0):
#     model.cuda()

# detr model
model, postprocessor = torch.hub.load('/root/auto-tmp/facebookresearch_detr_main', 'detr_resnet101_panoptic', source="local", pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()

input = "data/coco/val2017/000000001353.png"
im = Image.open(input)
img = transform(im).unsqueeze(0)
out = model(img)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

import io
# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
# We retrieve the ids corresponding to each mask
panoptic_seg.save("sample_detr.png")

# def generate_panoptic_image(image_path):
#     im = cv2.imread(image_path)
#     img = np.transpose(im, (2, 0, 1))
#     img_tensor = torch.from_numpy(img)
#     outputs = model([{"image": img_tensor}])
#     # print(outputs[0]["panoptic_seg"])
#     v = Visualizer(
#         im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
#     )
#     out = v.draw_panoptic_seg(
#         outputs[0]["panoptic_seg"][0].to("cpu"), outputs[0]["panoptic_seg"][1]
#     )

#     return out.get_image()

def generate_panoptic_image_detr(image_path):
    im = cv2.imread(image_path)
    img = np.transpose(im, (2, 0, 1))
    img_tensor = torch.from_numpy(img)
    outputs = model([{"image": img_tensor}])
    # print(outputs[0]["panoptic_seg"])
    

    return out.get_image()


with open("data/psg/psg_cls_basic.json", "r") as all_data_file:
    all_data = json.load(all_data_file)["data"]

    # pbar = tqdm.tqdm(all_data)
    # for data in pbar:
    #     input_path = f"data/coco/{data['file_name']}"
    #     out = generate_panoptic_image(input_path)
    #     cv2.imwrite(f"data/detr2/{data['file_name']}", out)
    #     # cv2.waitKey()

    input = f"data/coco/{all_data[0]['file_name']}"
    # out = generate_panoptic_image(input)
    # print(out)
