from configparser import ConfigParser
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import resize
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Any, Tuple, Union
import onnxruntime as ort
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


def preprocess(x, img_size):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

    return x


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


class SamEncoder:
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)
    

class SamDecoder:
    def __init__(
        self, model_path: str, device: str = "cpu", target_size: int = 1024, mask_threshold: float = 0.0, **kwargs
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError("Invalid device, please use 'cuda' or 'cpu' device.")

        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(model_path, opt, providers=provider, **kwargs)

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

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(*origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError("Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

        if boxes is not None:
            boxes = self.apply_boxes(boxes, origin_image_size, input_size).astype(np.float32)
            box_label = np.array([[2, 3] for _ in range(boxes.shape[0])], dtype=np.float32).reshape((-1, 2))
            point_coords = boxes
            point_labels = box_label

        input_dict = {"image_embeddings": img_embeddings, "point_coords": point_coords, "point_labels": point_labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = mask_postprocessing(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


class SAM_ONNX:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.arch = self.config.get('SAM', 'arch')
        self.engine = self.config.get('SAM', 'engine')
        self.encoder_path = self.config.get('SAM', 'encoder_path')
        self.decoder_path = self.config.get('SAM', 'decoder_path')
        self.cuda = self.config.getboolean('GPU', 'use_gpu')
        if self.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.encoder = SamEncoder(model_path=self.encoder_path, device=self.device)
        self.decoder = SamDecoder(model_path=self.decoder_path, device=self.device)

    def mask_predict(self, image, boxes):
        origin_image_size = image.shape[:2]
        if self.arch in ["l0", "l1", "l2"]:
            img = preprocess(image, img_size=512)
        elif self.arch in ["xl0", "xl1"]:
            img = preprocess(image, img_size=1024)

        img_embeddings = self.encoder(img)

        self.boxes = []

        for b in boxes:
            t = [int(b[i]) for i in range(4)]
            self.boxes.append(t)

        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=np.array(self.boxes),
        )

        if self.device == 'cpu':
            mask_list = [masks[i].numpy()[0][int(boxes[i][1]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][2])].astype(np.uint8) * 255 for i in range(len(masks))]
        else:
            mask_list = [masks[i].cpu().numpy()[0][int(boxes[i][1]):int(boxes[i][3]),int(boxes[i][0]):int(boxes[i][2])].astype(np.uint8) * 255 for i in range(len(masks))]

        return mask_list
