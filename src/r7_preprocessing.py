"""Laterality-preserving frozen image preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps


def letterbox_rgb(image: Image.Image, size: int = 224) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    side = max(width, height)
    horizontal = side - width
    vertical = side - height
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    squared = ImageOps.expand(
        image, border=(left, top, right, bottom), fill=(0, 0, 0)
    )
    return squared.resize(
        (size, size), resample=Image.Resampling.BILINEAR, reducing_gap=None
    )


def preprocess_image(path: str | Path, config: dict[str, Any]) -> torch.Tensor:
    preprocessing = config["preprocessing"]
    with Image.open(path) as image:
        resized = letterbox_rgb(image, int(preprocessing["input_size"]))
        array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(preprocessing["mean"], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(preprocessing["std"], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std
