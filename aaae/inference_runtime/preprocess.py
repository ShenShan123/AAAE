from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGENET_MEAN_HWC = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD_HWC = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

CLASSIFICATION_PROFILES: dict[str, dict[str, float | int | str]] = {
    "resnet50": {
        "kind": "center_crop",
        "crop_size": 224,
        "resize_short": 232,
        "norm": "imagenet",
        "resample": "bilinear",
    },
    "mobilenetv3": {
        "kind": "center_crop",
        "crop_size": 224,
        "resize_short": 232,
        "norm": "imagenet",
        "resample": "bilinear",
    },
    "vit_base_patch16_224": {
        "kind": "center_crop",
        "crop_size": 224,
        "resize_short": 248,
        "norm": "inception",
        "resample": "bilinear",
    },
    "deit_base_patch16_224": {
        "kind": "torchvision_center_crop",
        "crop_size": 224,
        "resize_short": 256,
        "norm": "imagenet",
        "resample": "bicubic",
    },
    "swin_tiny_patch4_window7_224": {
        "kind": "center_crop",
        "crop_size": 224,
        "resize_short": 248,
        "norm": "imagenet",
        "resample": "bilinear",
    },
    "yolov8s_cls": {
        "kind": "center_crop",
        "crop_size": 224,
        "resize_short": 224,
        "norm": "zero_one",
        "resample": "bilinear",
    },
    "efficientnet_lite4": {
        "kind": "efficientnet_lite",
        "crop_size": 300,
        "crop_fraction": 0.875,
        "norm": "lite_inception",
        "resample": "bilinear",
    },
}


def _resample(name: str) -> Image.Resampling:
    if name == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.BILINEAR


def center_crop_resize(image: Image.Image, crop_size: int, resize_short: int, resample: Image.Resampling) -> Image.Image:
    width, height = image.size
    scale = resize_short / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), resample)
    left = max((resized.width - crop_size) // 2, 0)
    top = max((resized.height - crop_size) // 2, 0)
    return resized.crop((left, top, left + crop_size, top + crop_size))


def torchvision_center_crop_resize(
    image: Image.Image,
    crop_size: int,
    resize_short: int,
    resample: Image.Resampling,
) -> Image.Image:
    """Match torchvision Resize(short_edge) followed by CenterCrop for PIL input."""
    width, height = image.size
    if min(width, height) != resize_short:
        if width < height:
            resized_width = resize_short
            resized_height = int(resize_short * height / width)
        else:
            resized_width = int(resize_short * width / height)
            resized_height = resize_short
        image = image.resize((resized_width, resized_height), resample)
    left = int(round((image.width - crop_size) / 2.0))
    top = int(round((image.height - crop_size) / 2.0))
    return image.crop((left, top, left + crop_size, top + crop_size))


def efficientnet_lite_center_crop(
    image: Image.Image,
    image_size: int,
    crop_fraction: float = 0.875,
    resample: Image.Resampling = Image.Resampling.BICUBIC,
) -> Image.Image:
    image_width, image_height = image.size
    crop_padding = round(image_size * (1.0 / crop_fraction - 1.0))
    padded_center_crop_size = int((image_size / (image_size + crop_padding)) * min(image_width, image_height))
    offset_height = max((image_height - padded_center_crop_size + 1) // 2, 0)
    offset_width = max((image_width - padded_center_crop_size + 1) // 2, 0)
    cropped = image.crop(
        (
            offset_width,
            offset_height,
            offset_width + padded_center_crop_size,
            offset_height + padded_center_crop_size,
        )
    )
    return cropped.resize((image_size, image_size), resample)


def normalize_hwc(array: np.ndarray, norm: str) -> np.ndarray:
    if norm == "imagenet":
        array = array / 255.0
        return (array - IMAGENET_MEAN_HWC) / IMAGENET_STD_HWC
    if norm == "inception":
        return (array / 255.0) * 2.0 - 1.0
    if norm == "lite_inception":
        return (array - 127.0) / 128.0
    if norm == "zero_one":
        return array / 255.0
    raise KeyError(f"unknown normalization profile: {norm}")


def prepare_classification_input(model_name: str, image_path: Path, nhwc: bool) -> np.ndarray:
    profile = CLASSIFICATION_PROFILES[model_name]
    image = Image.open(image_path).convert("RGB")
    kind = str(profile["kind"])
    crop_size = int(profile["crop_size"])
    resample = _resample(str(profile.get("resample", "bilinear")))
    if kind == "center_crop":
        image = center_crop_resize(
            image,
            crop_size=crop_size,
            resize_short=int(profile["resize_short"]),
            resample=resample,
        )
    elif kind == "torchvision_center_crop":
        image = torchvision_center_crop_resize(
            image,
            crop_size=crop_size,
            resize_short=int(profile["resize_short"]),
            resample=resample,
        )
    elif kind == "efficientnet_lite":
        image = efficientnet_lite_center_crop(
            image,
            image_size=crop_size,
            crop_fraction=float(profile["crop_fraction"]),
            resample=resample,
        )
    else:
        raise KeyError(f"unknown preprocessing kind: {kind}")
    array = np.asarray(image).astype(np.float32)
    array = normalize_hwc(array, str(profile["norm"]))
    if nhwc:
        return np.expand_dims(array, 0).astype(np.float32)
    chw = np.transpose(array, (2, 0, 1))
    return np.expand_dims(chw, 0).astype(np.float32)
