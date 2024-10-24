from configparser import ConfigParser
import tensorrt as trt
from torch2trt import TRTModule
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from typing import Tuple
import numpy as np
from copy import deepcopy


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def preprocess(x, img_size, device):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x).to(device)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0)

    return x


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords, original_size, new_size):
    old_h, old_w = original_size
    new_h, new_w = new_size
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes, original_size, new_size):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
    return boxes


def resize_longest_image_size(input_image_size: torch.Tensor, longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)

    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
    return masks


class SAM_TRT:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.arch = self.config.get('SAM', 'arch')
        self.engine = self.config.get('SAM', 'engine')
        self.encoder_path = self.config.get('SAM', 'encoder_path')
        self.decoder_path = self.config.get('SAM', 'decoder_path')

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(self.encoder_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.encoder = TRTModule(engine, input_names=["input_image"], output_names=["image_embeddings"])

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(self.decoder_path, "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.decoder = TRTModule(
            engine,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"],
        )

    def mask_predict(self, image, boxes):
        origin_image_size = image.shape[:2]
        if self.arch in ["l0", "l1", "l2"]:
            img = preprocess(image, img_size=512, device="cuda")
        elif self.arch in ["xl0", "xl1"]:
            img = preprocess(image, img_size=1024, device="cuda")

        image_embedding = self.encoder(img)
        image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

        input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

        self.boxes = []

        for b in boxes:
            t = [int(b[i]) for i in range(4)]
            self.boxes.append(t)

        self.boxes = np.array(self.boxes)

        self.boxes = apply_boxes(self.boxes, origin_image_size, input_size).astype(np.float32)
        self.box_label = np.array([[2, 3] for _ in range(self.boxes.shape[0])], dtype=np.float32).reshape((-1, 2))
        point_coords = self.boxes
        point_labels = self.box_label

        inputs = (image_embedding, torch.from_numpy(point_coords).to("cuda"), torch.from_numpy(point_labels).to("cuda"))
        assert all([x.dtype == torch.float32 for x in inputs])

        low_res_masks, _ = self.decoder(*inputs)
        low_res_masks = low_res_masks.reshape(1, -1, 256, 256)

        masks = mask_postprocessing(low_res_masks, origin_image_size)[0]
        masks = masks > 0.0
        masks = masks.cpu().numpy()

        mask_list = [masks[i][int(boxes[i][1]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][2])].astype(np.uint8) * 255 for i in range(len(masks))]

        return mask_list
    
