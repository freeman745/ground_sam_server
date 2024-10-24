import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict, load_model
import groundingdino.datasets.transforms as T

from configparser import ConfigParser

from utils import filter_bboxes_by_overlap, filter_large_small_bboxes


class GroundingDINO:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.CHECKPOINT_PATH = self.config.get('DINO', 'path')
        self.cuda = self.config.getboolean('GPU', 'use_gpu')
        if self.cuda:
            self.device = 'cuda:'+self.config.get('GPU', 'device')
        else:
            self.device = 'cpu'

        self.gd_model = load_model(self.CONFIG_PATH, self.CHECKPOINT_PATH, device=self.device)
        self.transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            )
        self.max_ratio = float(self.config.get('Model_config', 'max_ratio'))
        self.min_ratio = float(self.config.get('Model_config', 'min_ratio'))

    def bbox_predict(self, image, TEXT_PROMPT, BOX_TRESHOLD=0.2, TEXT_TRESHOLD=0.2):
        self.img = Image.fromarray(image).convert("RGB")
        self.img_s = np.asarray(self.img)
        self.image_transformed, _ = self.transform(self.img, None)
        h, w, _ = self.img_s.shape
        boxes, logits, phrases = predict(
            model=self.gd_model, 
            image=self.image_transformed, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD,
            device=self.device
        )
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        bbox_list = [box.tolist()+[logits[i].tolist()] for i, box in enumerate(xyxy)]

        bbox_list, phrases = filter_large_small_bboxes(bbox_list, phrases, self.max_ratio, self.min_ratio, image.shape[1], image.shape[0])
        bbox_list, phrases = filter_bboxes_by_overlap(bbox_list, phrases)
        
        return bbox_list, phrases
